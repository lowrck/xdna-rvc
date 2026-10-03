#!/usr/bin/env python3
"""Objective quality checks for a converted file against its source.

Measures, without listening:
  * pitch tracking  - RMVPE F0 of output vs input; median shift (semitones) and the share
                      of voiced frames within 0.5 semitone of the requested shift
  * voicing         - fraction of frames voiced in input/output
  * alignment       - lag (10 ms frames) maximising envelope cross-correlation
  * loudness        - RMS of input/output, clipping, NaN/Inf
  * discontinuities - largest sample-to-sample jump relative to the signal (click detector)
  * intelligibility - optional (--asr): word error rate of wav2vec2-base-960h transcriptions
                      against --reference text, for input and output

Use it to compare backends or precisions:
  python tools/evaluate_conversion.py in.wav out_cpu.wav --pitch 4 --rmvpe assets/rmvpe/rmvpe.pt
  python tools/evaluate_conversion.py in.wav out_npu_bf16.wav --pitch 4 --rmvpe ... --compare out_cpu.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from xdna_rvc_tools.audio import load_wav, resample


def envelope(a: np.ndarray, sr: int) -> np.ndarray:
    a = resample(a, sr, 16000)
    n = len(a) // 160
    return np.sqrt((a[: n * 160].reshape(n, 160) ** 2).mean(axis=1))


def best_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 30) -> int:
    m = min(len(x), len(y))
    x, y = x[:m] - x[:m].mean(), y[:m] - y[:m].mean()
    lags = list(range(-max_lag, max_lag + 1))
    cc = [np.dot(x[max(0, l):m + min(0, l)], y[max(0, -l):m - max(0, l)]) for l in lags]
    return lags[int(np.argmax(cc))]


def wer(ref: str, hyp: str) -> float:
    r, h = ref.upper().split(), hyp.upper().split()
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0], d[0, :] = range(len(r) + 1), range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def evaluate(inp: Path, out: Path, pitch: float, rmvpe_path: Path | None, reference: str | None, asr: bool,
             compare_to: Path | None) -> dict:
    x, sx = load_wav(inp)
    y, sy = load_wav(out)
    res: dict = {
        "input": str(inp), "output": str(out),
        "duration_in_s": len(x) / sx, "duration_out_s": len(y) / sy,
        "rms_in": float(np.sqrt(np.mean(x ** 2))), "rms_out": float(np.sqrt(np.mean(y ** 2))),
        "peak_out": float(np.abs(y).max()), "finite": bool(np.isfinite(y).all()),
        "clipped_fraction": float(np.mean(np.abs(y) >= 0.999)),
        "alignment_lag_frames": best_lag(envelope(x, sx), envelope(y, sy)),
    }
    dy = np.abs(np.diff(y))
    res["max_jump_over_p999"] = float(dy.max() / max(np.percentile(dy, 99.9), 1e-9))

    if rmvpe_path:
        import convert_rmvpe
        _, rm = convert_rmvpe.load_upstream(rmvpe_path, False)
        fx = rm.infer_from_audio(resample(x, sx, 16000), 0.03)
        fy = rm.infer_from_audio(resample(y, sy, 16000), 0.03)
        n = min(len(fx), len(fy))
        fx, fy = fx[:n], fy[:n]
        both = (fx > 0) & (fy > 0)
        semis = 12 * np.log2(fy[both] / fx[both]) if both.any() else np.array([np.nan])
        res.update({"voiced_in": float((fx > 0).mean()), "voiced_out": float((fy > 0).mean()),
                    "pitch_shift_median_st": float(np.nanmedian(semis)),
                    "pitch_within_half_semitone": float(np.mean(np.abs(semis - pitch) < 0.5))})

    if compare_to:
        z, sz = load_wav(compare_to)
        z = resample(z, sz, sy)
        m = min(len(y), len(z))
        ey, ez = envelope(y[:m], sy), envelope(z[:m], sy)
        res["compare_to"] = str(compare_to)
        res["envelope_correlation_vs_compare"] = float(np.corrcoef(ey, ez)[0, 1])
        spec = lambda a: np.log(np.abs(np.fft.rfft(a[: m - m % 1024].reshape(-1, 1024) * np.hanning(1024))) + 1e-6)
        res["log_spectral_distance_db_vs_compare"] = float(
            np.mean(np.sqrt(np.mean((20 / np.log(10) * (spec(y) - spec(z))) ** 2, axis=1))))

    if asr:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval()

        def transcribe(a, sr):
            iv = proc(resample(a, sr, 16000), sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                return proc.batch_decode(model(iv).logits.argmax(-1))[0]

        res["asr_input"] = transcribe(x, sx)
        res["asr_output"] = transcribe(y, sy)
        ref = reference or res["asr_input"]
        res["wer_input"] = wer(ref, res["asr_input"])
        res["wer_output"] = wer(ref, res["asr_output"])
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--pitch", type=float, default=0.0, help="pitch shift used for the conversion (semitones)")
    p.add_argument("--rmvpe", type=Path, help="rmvpe.pt for pitch analysis")
    p.add_argument("--asr", action="store_true", help="compute WER with wav2vec2-base-960h (downloads ~360 MB)")
    p.add_argument("--reference", help="reference transcript (default: ASR of the input)")
    p.add_argument("--compare", type=Path, help="another conversion of the same input to compare against")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    res = evaluate(a.input, a.output, a.pitch, a.rmvpe, a.reference, a.asr, a.compare)
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        for k, v in res.items():
            print(f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
