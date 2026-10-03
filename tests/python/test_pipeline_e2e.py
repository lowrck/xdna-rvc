"""End-to-end checks of the native pipeline against the Python/upstream references.

Need: built xdna-rvc-cli, converted shared models (content encoder, RMVPE) and
test_assets (rmvpe.pt, speech, a test voice). Skipped otherwise.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture(scope="module")
def voice_model(root) -> Path:
    p = root / "models" / "vctk_v2_f0_40k" / "model.json"
    if not p.is_file():
        pytest.skip("test voice not converted (see docs/model_conversion.md, 'test voices')")
    return p


def run_cli(cli, *args):
    r = subprocess.run([str(cli), "--no-log-file", "--log-level", "warning", *map(str, args)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def test_cpp_rmvpe_matches_upstream(cli, rmvpe_path, speech_wav, voice_model, tmp_path):
    import convert_rmvpe
    f0_path, audio_path = tmp_path / "f0.npy", tmp_path / "a.npy"
    run_cli(cli, "f0", speech_wav, "--model", voice_model, "-o", f0_path, "--dump-audio", audio_path)
    f0_cpp = np.load(f0_path)
    _, upstream = convert_rmvpe.load_upstream(rmvpe_path, False)
    f0_ref = upstream.infer_from_audio(np.load(audio_path), thred=0.03)
    assert f0_cpp.shape == f0_ref.shape
    voiced_agree = np.mean((f0_cpp > 0) == (f0_ref > 0))
    both = (f0_cpp > 0) & (f0_ref > 0)
    cents = 1200 * np.abs(np.log2(f0_cpp[both] / f0_ref[both]))
    assert voiced_agree > 0.995, voiced_agree
    assert np.percentile(cents, 99) < 5.0, cents.max()


@pytest.mark.parametrize("pitch", [0, 5])
def test_offline_conversion_tracks_pitch_and_stays_aligned(cli, rmvpe_path, speech_wav, voice_model, tmp_path, pitch):
    import evaluate_conversion
    out = tmp_path / "out.wav"
    report = tmp_path / "report.json"
    run_cli(cli, "convert", speech_wav, out, "--model", voice_model, "--backend", "cpu", "--seed", 1,
            f"--pitch={pitch}", "--report", report)
    res = evaluate_conversion.evaluate(speech_wav, out, pitch, rmvpe_path, None, False, None)
    assert res["finite"]
    assert abs(res["duration_in_s"] - res["duration_out_s"]) < 0.01
    assert abs(res["alignment_lag_frames"]) <= 1, res["alignment_lag_frames"]
    assert 0.2 * res["rms_in"] < res["rms_out"] < 5 * res["rms_in"]
    assert res["clipped_fraction"] < 1e-4
    assert abs(res["pitch_shift_median_st"] - pitch) < 0.3, res
    assert res["pitch_within_half_semitone"] > 0.85, res
    stages = {s["stage"]: s for s in json.loads(report.read_text())["stages"]}
    assert set(stages) == {"content_encoder", "rmvpe", "generator"}
    for s in stages.values():
        assert s["effective_backend"] == "cpu"
        assert s["provider_usage"][0]["provider"] == "CPUExecutionProvider"


def test_same_seed_is_deterministic(cli, speech_wav, voice_model, tmp_path):
    import soundfile as sf
    outs = []
    for i in range(2):
        out = tmp_path / f"o{i}.wav"
        run_cli(cli, "convert", speech_wav, out, "--model", voice_model, "--backend", "cpu", "--seed", 7)
        outs.append(sf.read(out)[0])
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-5)


def test_missing_generator_variant_gives_actionable_error(cli, speech_wav, voice_model, tmp_path):
    r = subprocess.run([str(cli), "--no-log-file", "convert", str(speech_wav), str(tmp_path / "x.wav"), "--model",
                        str(voice_model), "--block-ms", "70"], capture_output=True, text=True)
    assert r.returncode != 0
    assert "no generator exported" in r.stderr and "convert_rvc.py" in r.stderr
