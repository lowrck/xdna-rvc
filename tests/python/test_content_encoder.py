"""Content encoder: module correctness, ONNX export numerics and C++ parity."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch

from xdna_rvc_tools import hubert
from xdna_rvc_tools.onnx_export import export_onnx, ort_optimize_basic, validate_against_torch
from xdna_rvc_tools.validation import compare


@pytest.mark.parametrize("samples", [400, 720, 16000, 16001, 24320])
def test_frame_count_formula(samples: int) -> None:
    m = hubert.HubertContentEncoder(output_layer=1).eval()
    with torch.no_grad():
        y = m(torch.zeros(1, samples))
    assert y.shape[1] == hubert.num_frames(samples)


def test_export_random_weights_matches_torch(tmp_path) -> None:
    """Export path works without downloaded weights (random init, 2 layers)."""
    torch.manual_seed(0)
    m = hubert.HubertContentEncoder(output_layer=2, final_proj=True).eval()
    raw, final = tmp_path / "raw.onnx", tmp_path / "enc.onnx"
    export_onnx(m, (torch.randn(1, 8000) * 0.1,), raw, ["audio"], ["features"],
                dynamic_axes={"audio": {1: "samples"}, "features": {1: "frames"}})
    ort_optimize_basic(raw, final)
    rng = np.random.default_rng(0)
    for n in (3200, 12345):
        audio = (rng.standard_normal((1, n)) * 0.1).astype(np.float32)
        comps = validate_against_torch("rand", final, m, {"audio": audio}, 1e-3, 1e-4)
        assert all(c.passed for c in comps), [c.line() for c in comps]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_module_matches_transformers_reference(hubert_dir, version) -> None:
    pytest.importorskip("transformers")
    torch.manual_seed(1)
    audio = torch.randn(1, 16000) * 0.1
    model = hubert.build_content_encoder(hubert_dir, version)
    ref = hubert.compare_with_transformers(hubert_dir, version, audio)
    with torch.no_grad():
        ours = model(audio)
    c = compare(f"{version}_vs_transformers", ref.numpy(), ours.numpy(), 1e-3, 1e-4)
    assert c.passed, c.line()
    assert ours.shape[-1] == (256 if version == "v1" else 768)


def test_cpp_features_match_torch(cli, hubert_dir, shared_models, speech_wav, tmp_path) -> None:
    onnx_path = shared_models / "content_encoder_v2.onnx"
    if not onnx_path.is_file():
        pytest.skip("content_encoder_v2.onnx not converted")
    feats, audio = tmp_path / "f.npy", tmp_path / "a.npy"
    subprocess.run([str(cli), "--no-log-file", "--log-level", "warning", "features", str(speech_wav),
                    "--encoder", str(onnx_path), "--backend", "cpu", "-o", str(feats), "--dump-audio", str(audio)],
                   check=True, capture_output=True)
    a16 = np.load(audio)
    f_cpp = np.load(feats)
    model = hubert.build_content_encoder(hubert_dir, "v2")
    with torch.no_grad():
        f_ref = model(torch.from_numpy(a16)[None])[0].numpy()
    c = compare("cpp_vs_torch", f_ref, f_cpp, 1e-3, 1e-4)
    assert c.passed, c.line()
