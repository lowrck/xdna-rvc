#!/usr/bin/env python3
"""Convert RMVPE (rmvpe.pt) to ONNX for the native pitch extractor, with validation.

The ONNX model maps a log-mel spectrogram to pitch salience:
    mel     float32[1, 128, frames]   frames must be a multiple of 32
    hidden  float32[1, frames, 360]   sigmoid salience over 360 cent bins
The mel front end (STFT n_fft=1024, hop 160, Hann, center/reflect, 128 HTK mel bands
30-8000 Hz, log(clamp(x, 1e-5))) runs natively on the CPU in the C++ runtime. The exact
librosa filterbank used by upstream RVC is exported next to the model as
rmvpe_mel_basis.npy so both sides use identical numbers.

Validation compares, on identical inputs:
  * ONNX Runtime vs the upstream RMVPE network (salience)
  * the full upstream RMVPE.infer_from_audio() vs (upstream mel -> ONNX -> upstream decode) f0

Example:
  python tools/convert_rmvpe.py --rmvpe assets/rmvpe/rmvpe.pt --out-dir models/shared \
      --validate-wav test.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from xdna_rvc_tools import CONVERSION_VERSION
from xdna_rvc_tools.audio import load_wav_mono_16k
from xdna_rvc_tools.onnx_export import export_onnx, op_histogram, ort_optimize_basic, run_ort, validate_against_torch
from xdna_rvc_tools.safe_load import load_checkpoint, sha256_file
from xdna_rvc_tools.validation import compare, write_report

log = logging.getLogger("convert_rmvpe")

MAX_ABS_TOL = 1e-3   # salience is in [0, 1]
REL_RMS_TOL = 1e-4


def load_upstream(path: Path, allow_unsafe: bool):
    from xdna_rvc_tools.rvc.upstream import rmvpe as up

    sd = load_checkpoint(path, allow_unsafe=allow_unsafe)
    model = up.E2E(4, 1, (2, 2))
    model.load_state_dict(sd, strict=True)
    model = model.float().eval()
    wrapper = up.RMVPE.__new__(up.RMVPE)  # full upstream helper, without re-loading weights
    wrapper.is_half = False
    wrapper.device = "cpu"
    wrapper.resample_kernel = {}
    wrapper.mel_extractor = up.MelSpectrogram(False, 128, 16000, 1024, 160, None, 30, 8000)
    wrapper.model = model
    cents = 20 * np.arange(360) + 1997.3794084376191
    wrapper.cents_mapping = np.pad(cents, (4, 4))
    return model, wrapper


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rmvpe", type=Path, required=True, help="rmvpe.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--validate-wav", type=Path)
    p.add_argument("--exporter", choices=["torchscript", "dynamo"], default="torchscript")
    p.add_argument("--allow-unsafe-pickle", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    _bootstrap.setup_logging(args.verbose)

    model, upstream = load_upstream(args.rmvpe, args.allow_unsafe_pickle)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / "rmvpe.raw.onnx"
    final = args.out_dir / "rmvpe.onnx"
    example = torch.randn(1, 128, 64)
    export_onnx(model, (example,), raw, ["mel"], ["hidden"],
                dynamic_axes={"mel": {2: "frames"}, "hidden": {1: "frames"}}, exporter=args.exporter,
                metadata={"stage": "pitch", "method": "rmvpe", "sample_rate": 16000, "hop": 160, "n_fft": 1024,
                          "n_mels": 128, "fmin": 30, "fmax": 8000, "frame_multiple": 32, "threshold": 0.03,
                          "source_sha256": sha256_file(args.rmvpe)})
    ort_optimize_basic(raw, final)
    raw.unlink()
    mel_basis = upstream.mel_extractor.mel_basis.numpy().astype(np.float32)
    np.save(args.out_dir / "rmvpe_mel_basis.npy", mel_basis)
    log.info("exported %s; mel basis %s; ops: %s", final, mel_basis.shape, op_histogram(final))

    comps = []
    rng = np.random.default_rng(7)
    # Realistic log-mel range is roughly [-11.5, 3].
    for frames in (32, 128, 320):
        mel = (rng.standard_normal((1, 128, frames)) * 2.0 - 5.0).astype(np.float32)
        comps += validate_against_torch(f"rmvpe/noise_{frames}", final, model, {"mel": mel}, MAX_ABS_TOL,
                                        REL_RMS_TOL, ["hidden"])

    if args.validate_wav:
        audio = load_wav_mono_16k(args.validate_wav)
        with torch.no_grad():
            mel = upstream.extract_mel(audio, center=True)
        n = mel.shape[-1]
        pad = 32 * ((n - 1) // 32 + 1) - n
        mel_p = torch.nn.functional.pad(mel, (0, pad)).numpy().astype(np.float32)
        comps += validate_against_torch("rmvpe/speech", final, model, {"mel": mel_p}, MAX_ABS_TOL, REL_RMS_TOL,
                                        ["hidden"])
        f0_ref = upstream.infer_from_audio(audio, thred=0.03)
        hidden = run_ort(final, {"mel": mel_p})[0][0, :n]
        f0_onnx = upstream.decode(hidden, thred=0.03)
        voiced_ref, voiced_onnx = f0_ref > 0, f0_onnx > 0
        agree = float(np.mean(voiced_ref == voiced_onnx))
        both = voiced_ref & voiced_onnx
        cents = 1200 * np.abs(np.log2(f0_onnx[both] / f0_ref[both])) if both.any() else np.zeros(1)
        c = compare("rmvpe/speech_f0_hz", f0_ref, f0_onnx, 1.0, 1e-3)
        c.notes.append(f"voicing agreement {agree:.4f}, max cents error {cents.max():.3f}")
        c.passed = c.passed or (agree > 0.995 and cents.max() < 5.0)
        log.info("%s; %s", c.line(), c.notes[-1])
        comps.append(c)
        np.save(args.out_dir / "rmvpe.validation_f0.npy", f0_ref.astype(np.float32))

    report = args.out_dir / "rmvpe.validation.json"
    write_report(report, comps, {"model": str(final), "conversion_version": CONVERSION_VERSION})
    ok = all(c.passed for c in comps)
    log.info("validation %s -> %s", "PASSED" if ok else "FAILED", report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
