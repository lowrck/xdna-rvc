#!/usr/bin/env python3
"""Convert the RVC HuBERT/ContentVec content encoder to ONNX, with validation.

Input: the HuBERT checkpoint used by RVC, either
  * the Transformers directory from upstream RVC (assets/hubert_base: config.json +
    pytorch_model.bin), or
  * a fairseq checkpoint (hubert_base.pt) - needs --allow-unsafe-pickle, because
    fairseq checkpoints embed pickled config objects.

Output (per RVC feature version):
  content_encoder_v1.onnx   audio[1, samples] -> features[1, frames, 256]  (layer 9 + final_proj)
  content_encoder_v2.onnx   audio[1, samples] -> features[1, frames, 768]  (layer 12)
  content_encoder_vX.validation.json

Every export is validated: identical inputs (noise + optional real speech) go through
PyTorch and ONNX Runtime and max/mean/relative errors are reported. The script exits
non-zero if validation fails.

Example:
  python tools/convert_contentvec.py --hubert assets/hubert_base --out-dir models/shared \
      --version v1 --version v2 --validate-wav test.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import _bootstrap  # noqa: F401  (sys.path setup)
import numpy as np
import torch

from xdna_rvc_tools import CONVERSION_VERSION, hubert
from xdna_rvc_tools.audio import load_wav_mono_16k
from xdna_rvc_tools.onnx_export import export_onnx, op_histogram, ort_optimize_basic, validate_against_torch
from xdna_rvc_tools.safe_load import sha256_file
from xdna_rvc_tools.validation import write_report

log = logging.getLogger("convert_contentvec")

# FP32 export of the same FP32 math: errors come only from op ordering/fusion.
MAX_ABS_TOL = 1e-3
REL_RMS_TOL = 1e-4


def checkpoint_hash(path) -> str:
    from pathlib import Path
    p = Path(path)
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            if (p / name).is_file():
                return sha256_file(p / name)
    return sha256_file(p)


def convert(args: argparse.Namespace, version: str) -> bool:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = hubert.build_content_encoder(args.hubert, version, allow_unsafe=args.allow_unsafe_pickle)
    log.info("[%s] built content encoder (%d layers, %d-dim) in %.1f s", version, model.output_layer,
             model.feature_dim, time.perf_counter() - t0)

    example = torch.randn(1, 16000) * 0.1
    raw_path = out_dir / f"content_encoder_{version}.raw.onnx"
    final_path = out_dir / f"content_encoder_{version}.onnx"
    export_onnx(
        model, (example,), raw_path,
        input_names=["audio"], output_names=["features"],
        dynamic_axes={"audio": {1: "samples"}, "features": {1: "frames"}},
        exporter=args.exporter,
        metadata={
            "stage": "content_encoder",
            "rvc_version": version,
            "feature_dim": model.feature_dim,
            "sample_rate": hubert.SAMPLE_RATE,
            "hop": hubert.HOP,
            "source_sha256": checkpoint_hash(args.hubert),
        },
    )
    ort_optimize_basic(raw_path, final_path)
    raw_path.unlink()
    log.info("[%s] exported %s; ops: %s", version, final_path, op_histogram(final_path))

    # Validation: several lengths of noise, plus real speech if provided.
    rng = np.random.default_rng(1234)
    cases = {f"noise_{n}": (rng.standard_normal((1, n)) * 0.1).astype(np.float32) for n in (4000, 16000, 37120)}
    if args.validate_wav:
        speech = load_wav_mono_16k(args.validate_wav)[: 16000 * 8]
        cases["speech"] = speech[None, :].astype(np.float32)
    comparisons = []
    for name, audio in cases.items():
        comparisons += validate_against_torch(
            f"{version}/{name}", final_path, model, {"audio": audio}, MAX_ABS_TOL, REL_RMS_TOL, ["features"])
        frames = hubert.num_frames(audio.shape[1])
        log.debug("%s: %d samples -> %d frames", name, audio.shape[1], frames)

    if args.reference_transformers:
        # Also confirm our module equals the upstream RVC reference (transformers HubertModel).
        from xdna_rvc_tools.validation import compare
        audio = torch.from_numpy(cases["speech" if "speech" in cases else "noise_16000"])
        ref = hubert.compare_with_transformers(args.hubert, version, audio)
        with torch.no_grad():
            ours = model(audio)
        c = compare(f"{version}/module_vs_transformers", ref.numpy(), ours.numpy(), MAX_ABS_TOL, REL_RMS_TOL)
        log.info("%s", c.line())
        comparisons.append(c)

    report = out_dir / f"content_encoder_{version}.validation.json"
    write_report(report, comparisons, {"model": str(final_path), "conversion_version": CONVERSION_VERSION})
    ok = all(c.passed for c in comparisons)
    log.info("[%s] validation %s -> %s", version, "PASSED" if ok else "FAILED", report)
    return ok


def main() -> int:
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hubert", type=Path, required=True, help="HuBERT checkpoint (Transformers dir or .pt)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--version", action="append", choices=["v1", "v2"], help="RVC feature version(s) to export")
    p.add_argument("--validate-wav", type=Path, help="speech WAV used as an additional validation input")
    p.add_argument("--reference-transformers", action="store_true",
                   help="also compare against transformers.HubertModel (the upstream RVC implementation)")
    p.add_argument("--exporter", choices=["torchscript", "dynamo"], default="torchscript")
    p.add_argument("--allow-unsafe-pickle", action="store_true",
                   help="allow full pickle loading for checkpoints that fail weights_only loading (trusted files only)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    _bootstrap.setup_logging(args.verbose)
    versions = args.version or ["v2"]
    ok = True
    for v in versions:
        ok &= convert(args, v)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
