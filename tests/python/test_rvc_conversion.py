"""RVC checkpoint inspection, export-model equivalence, ONNX numerics and index export.

Most tests build randomly initialised synthesizers with the real RVC configurations,
so they run without downloading any voice model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

from xdna_rvc_tools.onnx_export import export_onnx, ort_optimize_basic, validate_against_torch
from xdna_rvc_tools.rvc.checkpoint import (KNOWN_CONFIGS, UnsupportedModelError, build_upstream_synthesizer,
                                           inspect_checkpoint, make_inference_checkpoint)
from xdna_rvc_tools.rvc.export_model import GeneratorExport, StreamGeometry, coarse_pitch, example_inputs, reference_infer
from xdna_rvc_tools.rvc.upstream import models as upstream_models
from xdna_rvc_tools.stream_config import PRESETS, StreamConfig, parse_stream
from xdna_rvc_tools.validation import compare

warnings.filterwarnings("ignore")


def random_checkpoint(tmp_path, version="v2", f0=True, sr=40000, n_spk=3, layout="inference", seed=0):
    torch.manual_seed(seed)
    cfg = list(KNOWN_CONFIGS[(version, sr)])
    cfg[-3] = n_spk
    cls = {("v1", True): upstream_models.SynthesizerTrnMs256NSFsid,
           ("v1", False): upstream_models.SynthesizerTrnMs256NSFsid_nono,
           ("v2", True): upstream_models.SynthesizerTrnMs768NSFsid,
           ("v2", False): upstream_models.SynthesizerTrnMs768NSFsid_nono}[(version, f0)]
    net = cls(*cfg, is_half=False)
    sd = net.state_dict()
    path = tmp_path / f"rand_{version}_{int(f0)}_{sr}_{layout}.pth"
    if layout == "inference":
        torch.save({"weight": {k: v.half() for k, v in sd.items() if not k.startswith("enc_q")}, "config": cfg,
                    "info": "random", "sr": f"{sr // 1000}k", "f0": int(f0), "version": version}, path)
    else:
        name = path.with_name(f"G_{sr // 1000}k.pth")
        torch.save({"model": {k: v.half() for k, v in sd.items()}, "iteration": 1, "learning_rate": 1e-4}, name)
        path = name
    return path


@pytest.mark.parametrize("version,f0,sr", [("v1", True, 40000), ("v1", False, 32000), ("v2", True, 48000),
                                           ("v2", False, 40000), ("v2", True, 32000), ("v1", True, 48000)])
def test_inspection_derives_architecture_from_weights(tmp_path, version, f0, sr):
    info, sd = inspect_checkpoint(random_checkpoint(tmp_path, version, f0, sr))
    assert info.version == version
    assert info.feature_dim == (256 if version == "v1" else 768)
    assert info.uses_f0 == f0
    assert info.sample_rate == sr
    assert info.n_speakers == 3
    assert info.upsample_factor * 100 == sr
    assert info.layout == "inference"
    assert not any(k.startswith("enc_q") for k in sd)


def test_training_checkpoint_config_is_reconstructed(tmp_path):
    p = random_checkpoint(tmp_path, "v1", True, 48000, layout="training")
    info, _ = inspect_checkpoint(p)
    assert info.layout == "training"
    assert info.sample_rate == 48000  # v1 32k and 48k share kernel sizes; spec_channels/file name disambiguate
    assert info.config["upsample_rates"] == [10, 6, 2, 2, 2]
    assert any("reconstructed" in w for w in info.warnings)


def test_metadata_disagreement_is_reported(tmp_path):
    p = random_checkpoint(tmp_path, "v2", True, 40000)
    ck = torch.load(p, weights_only=True)
    ck["version"] = "v1"  # lies: weights are 768-dim
    torch.save(ck, p)
    info, _ = inspect_checkpoint(p)
    assert info.version == "v2"
    assert any("metadata says version" in w for w in info.warnings)


def test_unsupported_vocoder_is_rejected(tmp_path):
    p = random_checkpoint(tmp_path, "v2", True, 40000)
    ck = torch.load(p, weights_only=True)
    ck["vocoder"] = "RefineGAN"
    torch.save(ck, p)
    with pytest.raises(UnsupportedModelError, match="RefineGAN"):
        inspect_checkpoint(p)


def test_non_rvc_checkpoint_is_rejected(tmp_path):
    p = tmp_path / "other.pth"
    torch.save({"weight": {"conv.weight": torch.zeros(3, 3)}}, p)
    with pytest.raises(UnsupportedModelError, match="not an RVC synthesizer"):
        inspect_checkpoint(p)


@pytest.mark.parametrize("version,f0", [("v2", True), ("v1", False)])
def test_export_model_equals_upstream_and_onnx_equals_torch(tmp_path, version, f0):
    info, sd = inspect_checkpoint(random_checkpoint(tmp_path, version, f0, 40000))
    upstream = build_upstream_synthesizer(info, sd)
    geom = StreamGeometry(frames=48, skip_head=30, return_length=12)  # flow_head = 6
    model = GeneratorExport(info, sd, geom).eval()
    x = example_inputs(model, seed=3)
    with torch.no_grad():
        ours = model(*x.values())
    ref = reference_infer(upstream, info, geom, x)
    c = compare("export_vs_upstream", ref.numpy(), ours.numpy(), 1e-4, 1e-4)
    assert c.passed, c.line()
    assert ours.shape == (1, geom.return_length * info.upsample_factor)

    raw, final = tmp_path / "g.raw.onnx", tmp_path / "g.onnx"
    export_onnx(model, tuple(x.values()), raw, list(x.keys()), ["audio"])
    ort_optimize_basic(raw, final)
    feeds = {k: v.numpy() for k, v in example_inputs(model, seed=4).items()}
    comps = validate_against_torch("onnx", final, model, feeds, 1e-3, 1e-4)
    assert all(c.passed for c in comps), [c.line() for c in comps]


def test_noise_inputs_are_used(tmp_path):
    """Guards against an export that silently ignores its noise inputs."""
    info, sd = inspect_checkpoint(random_checkpoint(tmp_path, "v2", True, 40000))
    model = GeneratorExport(info, sd, StreamGeometry(40, 10, 10)).eval()
    x = example_inputs(model, seed=1)
    with torch.no_grad():
        a = model(*x.values())
        x2 = dict(x, noise=torch.randn_like(x["noise"]))
        b = model(*x2.values())
        x3 = dict(x, rnd=torch.randn_like(x["rnd"]))
        c = model(*x3.values())
    assert not torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_coarse_pitch_matches_upstream_formula():
    # upstream infer/vc/pipeline.py get_f0(), numpy version
    f0 = np.array([0.0, 50.0, 100.0, 440.0, 1100.0, 2000.0])
    mel_min, mel_max = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel[mel > 0] = (mel[mel > 0] - mel_min) * 254 / (mel_max - mel_min) + 1
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    expected = np.rint(mel).astype(np.int64)
    assert coarse_pitch(torch.from_numpy(f0)[None]).tolist() == [expected.tolist()]


def test_stream_presets_and_parsing():
    g = StreamConfig(100, 40, 600).geometry()
    assert (g.frames, g.skip_head, g.return_length) == (60 + 4 + 1 + 10, 60, 10 + 4 + 1)
    g2 = StreamConfig(100, 80, 500).geometry()
    assert g2.return_length == 10 + 4 + 1  # SOLA buffer capped at 40 ms
    assert parse_stream("balanced") == PRESETS["balanced"]
    assert parse_stream("80,30,400").geometry().frames == 40 + 3 + 1 + 8
    # Same table as tests/cpp/test_rvc_components.cpp
    assert PRESETS["low_latency"].geometry().tag() == "T43_s30_r7"
    assert PRESETS["balanced"].geometry().tag() == "T45_s30_r9"
    assert PRESETS["quality"].geometry().tag() == "T75_s60_r15"
    assert parse_stream("40,20,300,60").geometry().frames == 30 + 2 + 1 + 4 + 6
    with pytest.raises(ValueError):
        parse_stream("85,40,600")


def test_index_export_matches_faiss(tmp_path):
    faiss = pytest.importorskip("faiss")
    from xdna_rvc_tools.rvc.index import blend, ivf_search, read_faiss_index, write_index_dir

    rng = np.random.default_rng(0)
    data = rng.standard_normal((4000, 32)).astype(np.float32)
    index = faiss.index_factory(32, "IVF40,Flat")
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(data)
    index.add(data)
    path = tmp_path / "t.index"
    faiss.write_index(index, str(path))

    arr = read_faiss_index(path)
    assert arr.nlist == 40 and arr.vectors.shape == (4000, 32)
    q = data[:50] + rng.standard_normal((50, 32)).astype(np.float32) * 0.1
    D, I = index.search(q, 8)
    d, rows = ivf_search(arr, q, 8)
    assert np.mean(arr.ids[rows] == I) > 0.99
    np.testing.assert_allclose(d, D, rtol=1e-4, atol=1e-4)

    out = write_index_dir(arr, tmp_path / "idx", path)
    meta = json.loads((out / "index.json").read_text())
    assert meta["dim"] == 32 and meta["ntotal"] == 4000
    assert np.load(out / "list_offsets.npy")[-1] == 4000

    mixed = blend(arr, q, 0.5)
    assert mixed.shape == q.shape
    np.testing.assert_allclose(blend(arr, q, 0.0), q)


def test_convert_rvc_cli_end_to_end(tmp_path, root):
    pth = random_checkpoint(tmp_path, "v2", True, 40000)
    out = tmp_path / "models"
    cmd = [sys.executable, str(root / "tools" / "convert_rvc.py"), str(pth), "--models-dir", str(out),
           "--name", "rand", "--stream", "60,30,200", "--preset", "low_latency"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    model = json.loads((out / "rand" / "model.json").read_text())
    assert model["rvc"]["version"] == "v2" and model["rvc"]["uses_f0"] is True
    assert len(model["generators"]) == 2
    for g in model["generators"]:
        assert (out / "rand" / g["path"]).is_file()
        assert all(v["passed"] for v in g["validation"].values())
    # second run reuses outputs
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode == 0 and "reusing" in r2.stderr
