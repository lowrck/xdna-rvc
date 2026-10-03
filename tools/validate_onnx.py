#!/usr/bin/env python3
"""Compare an ONNX model's outputs across execution providers or against another model file.

Typical uses
  # Is the static-shape copy identical to the dynamic export?
  python tools/validate_onnx.py models/shared/content_encoder_v2.onnx \
      --other models/shared/xdna/static/content_encoder_v2__samples12000.onnx --dim samples=12000
  # BF16 on the NPU vs FP32 on the CPU (Ryzen AI environment):
  python tools/validate_onnx.py model.onnx --provider VitisAIExecutionProvider \
      --vitis-config models/voice/xdna/vaiml_config.json --vitis-cache cache --vitis-key mykey
  # DirectML FP32/FP16 vs CPU:
  python tools/validate_onnx.py model.onnx --provider DmlExecutionProvider

Inputs are random with realistic ranges for the xdna-rvc models (audio, mel, phone,
pitch...). The reference is always FP32 on the CPU. Exit code 1 if the tolerance fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import onnxruntime as ort

from xdna_rvc_tools.validation import compare


def make_session(path: Path, provider: str, dims: dict[str, int], args) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    for k, v in dims.items():
        so.add_free_dimension_override_by_name(k, v)
    opts = {}
    if provider == "VitisAIExecutionProvider":
        opts = {"enable_cache_file_io_in_mem": "0"}
        if args.vitis_config:
            opts["config_file"] = str(Path(args.vitis_config).resolve())
        if args.vitis_cache:
            opts["cache_dir"] = str(Path(args.vitis_cache).resolve())
        if args.vitis_key:
            opts["cache_key"] = args.vitis_key
    return ort.InferenceSession(str(path), so, providers=[provider], provider_options=[opts])


def feeds_for(sess: ort.InferenceSession, dims: dict[str, int], rng) -> dict[str, np.ndarray]:
    out = {}
    for i in sess.get_inputs():
        shape = []
        for d in i.shape:
            if isinstance(d, int):
                shape.append(d)
            elif d in dims:
                shape.append(dims[d])
            else:
                raise SystemExit(f"input {i.name} has symbolic dim '{d}'; pass --dim {d}=N")
        if i.type == "tensor(int64)":
            out[i.name] = rng.integers(1, 255, size=shape).astype(np.int64) if i.name == "pitch" else np.zeros(shape, np.int64)
        elif i.name == "mel":
            out[i.name] = (rng.standard_normal(shape) * 2 - 5).astype(np.float32)
        elif i.name == "audio":
            out[i.name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        elif i.name == "pitchf":
            out[i.name] = rng.uniform(80, 400, size=shape).astype(np.float32)
        else:
            out[i.name] = rng.standard_normal(shape).astype(np.float32)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path)
    p.add_argument("--other", type=Path, help="second model file (default: same model)")
    p.add_argument("--provider", default="CPUExecutionProvider", help="provider for the candidate run")
    p.add_argument("--dim", action="append", default=[], help="symbolic dim override name=value")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--max-abs", type=float, default=1e-3)
    p.add_argument("--rel-rms", type=float, default=1e-4)
    p.add_argument("--vitis-config")
    p.add_argument("--vitis-cache")
    p.add_argument("--vitis-key")
    a = p.parse_args()
    dims = {k: int(v) for k, v in (d.split("=") for d in a.dim)}
    ref = make_session(a.model, "CPUExecutionProvider", dims, a)
    cand = make_session(a.other or a.model, a.provider, dims, a)
    print(f"reference: {a.model} on CPU (FP32); candidate: {a.other or a.model} on {cand.get_providers()[0]}")
    rng = np.random.default_rng(0)
    ok = True
    for t in range(a.trials):
        f = feeds_for(ref, dims, rng)
        for name, r, c in zip([o.name for o in ref.get_outputs()], ref.run(None, f), cand.run(None, f)):
            cmp = compare(f"trial{t}:{name}", r, c, a.max_abs, a.rel_rms)
            print(cmp.line())
            ok &= cmp.passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
