#!/usr/bin/env python3
"""Precompile a voice's models for the AMD XDNA 2 NPU (BF16) and validate against FP32.

Why this exists: per Ryzen AI 1.8 "Application Development", the deployment VitisAI EP
cannot compile BF16 models on the fly; applications must ship precompiled models. This
script must therefore run inside the Ryzen AI conda environment (`conda activate
ryzen-ai-1.8.0`) on a machine with an STX/KRK NPU.

For every stage and stream geometry it:
  1. writes a static-shape copy of the FP32 model (ORT basic optimisation with free
     dimension overrides, same procedure as the C++ runtime),
  2. compiles it with the VitisAI EP (BF16 via the VAIML compiler) into
     <cache_dir>/<cache_key>, writing the operator assignment report,
  3. runs identical inputs through the compiled NPU session and the FP32 CPU session
     and records the error (BF16 is not bit-exact; tolerances are per stage),
  4. records everything under "xdna" in the voice's model.json, which the runtime reads.

Precision policy: BF16 only. No INT8 quantisation here (see docs/implementation_plan.md).

Without an NPU (or outside the Ryzen AI environment) use --prepare-only: static models
and the manifest are written with "compiled": false, so the compile step can be finished
later on the target machine.

  python tools/compile_xdna.py models/myvoice/model.json --preset balanced --preset low_latency
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import onnxruntime as ort

from xdna_rvc_tools.stream_config import PRESETS, StreamConfig, parse_stream
from xdna_rvc_tools.validation import compare

log = logging.getLogger("compile_xdna")

VITIS = "VitisAIExecutionProvider"
REPORT_NAME = "vitisai_ep_report.json"

# Default BF16 compile configuration from the Ryzen AI 1.8 documentation ("Config File Options").
VAIML_CONFIG = {
    "passes": [
        {"name": "init", "plugin": "vaip-pass_init"},
        {"name": "vaiml_partition", "plugin": "vaip-pass_vaiml_partition",
         "vaiml_config": {"optimize_level": 1, "preferred_data_storage": "auto"}},
    ],
    "target": "VAIML",
    "targets": [{"name": "VAIML", "pass": ["init", "vaiml_partition"]}],
}

# BF16 has ~3 significant decimal digits; tolerances are on relative RMS error.
STAGE_TOLERANCE = {"content_encoder": 0.05, "rmvpe": 0.05, "generator": 0.10}


def sha256_prefix(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def make_static(src: Path, dims: dict[str, int], out_dir: Path) -> Path:
    tag = "__".join(f"{k}{v}" for k, v in sorted(dims.items()))
    out = out_dir / (f"{src.stem}__{tag}.onnx" if tag else f"{src.stem}.static.onnx")
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    if not dims:  # already static (generators): the export itself is used
        return src
    out_dir.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    # Pin dims only. Re-optimising (constant folding) the pinned graph yields files that
    # ORT 1.30 fails to load at ORT_ENABLE_ALL; see docs/troubleshooting.md.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for k, v in dims.items():
        so.add_free_dimension_override_by_name(k, v)
    so.optimized_model_filepath = str(out)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    return out


def example_feeds(sess: ort.InferenceSession, rng: np.random.Generator, stage: str) -> dict[str, np.ndarray]:
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        if i.type == "tensor(int64)":
            feeds[i.name] = (rng.integers(20, 200, size=shape) if i.name == "pitch"
                             else np.zeros(shape)).astype(np.int64)
        elif i.name == "audio":
            feeds[i.name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        elif i.name == "mel":
            feeds[i.name] = (rng.standard_normal(shape) * 2 - 5).astype(np.float32)
        elif i.name == "pitchf":
            feeds[i.name] = np.full(shape, 180.0, np.float32)
        else:
            feeds[i.name] = rng.standard_normal(shape).astype(np.float32) * (0.5 if i.name == "phone" else 1.0)
    return feeds


def compile_one(stage: str, static: Path, cache_dir: Path, cache_key: str, config: Path, prepare_only: bool) -> dict:
    entry: dict = {"compiled": False}
    if prepare_only:
        return entry
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = REPORT_NAME
    so = ort.SessionOptions()
    so.log_severity_level = 2
    opts = {"config_file": str(config.resolve()), "cache_dir": str(cache_dir.resolve()), "cache_key": cache_key,
            "enable_cache_file_io_in_mem": "0"}
    t0 = time.perf_counter()
    npu = ort.InferenceSession(str(static), so, providers=[VITIS], provider_options=[opts])
    entry["compile_seconds"] = round(time.perf_counter() - t0, 1)
    entry["compiled"] = True
    for candidate in (cache_dir / cache_key / REPORT_NAME, cache_dir / REPORT_NAME):
        if candidate.is_file():
            stats = {s["name"]: s.get("nodeNum", 0) for s in json.loads(candidate.read_text()).get("deviceStat", [])}
            entry["assignment"] = stats
            break
    cpu = ort.InferenceSession(str(static), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    comps = []
    for trial in range(3):
        feeds = example_feeds(cpu, rng, stage)
        ref = cpu.run(None, feeds)[0]
        out = npu.run(None, feeds)[0]
        c = compare(f"{stage}/bf16_vs_fp32_{trial}", ref, out, float("inf"), STAGE_TOLERANCE[stage])
        log.info("%s", c.line())
        comps.append(c)
    t1 = time.perf_counter()
    for _ in range(10):
        npu.run(None, feeds)
    entry["npu_run_ms"] = round((time.perf_counter() - t1) * 100, 2)
    entry["bf16_vs_fp32"] = {"max_rel_rms": max(c.rel_rms for c in comps), "min_cosine": min(c.cosine for c in comps),
                             "passed": all(c.passed for c in comps)}
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_json", type=Path)
    p.add_argument("--preset", action="append", choices=list(PRESETS))
    p.add_argument("--stream", action="append", default=[])
    p.add_argument("--stages", default="content_encoder,rmvpe,generator")
    p.add_argument("--prepare-only", action="store_true", help="write static models + manifest without compiling")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    _bootstrap.setup_logging(a.verbose)

    model_dir = a.model_json.parent
    model = json.loads(a.model_json.read_text())
    if not a.prepare_only and VITIS not in ort.get_available_providers():
        log.error("%s is not available in this Python environment (providers: %s). Run inside the Ryzen AI "
                  "conda environment on an STX/KRK machine, or use --prepare-only. See docs/amd_xdna_setup.md.",
                  VITIS, ort.get_available_providers())
        return 2

    configs: list[StreamConfig] = [PRESETS[x] for x in (a.preset or [])] + [parse_stream(s) for s in a.stream]
    if not configs:
        configs = [StreamConfig(**g["stream"]) for g in model["generators"]]
    stages = set(a.stages.split(","))
    xdna_dir = model_dir / "xdna"
    shared_xdna = (model_dir / model["content_encoder"]["path"]).parent / "xdna"
    for d in (xdna_dir, shared_xdna):
        d.mkdir(parents=True, exist_ok=True)
    config_file = xdna_dir / "vaiml_config.json"
    config_file.write_text(json.dumps(VAIML_CONFIG, indent=2))

    rel = lambda p: os.path.relpath(p, model_dir).replace(os.sep, "/")
    entries = []
    for cfg in configs:
        g = cfg.geometry()
        window = g.frames * 160
        block16 = cfg.block_ms * 16
        seg = 5120 * ((block16 + 800 - 1) // 5120 + 1) - 160
        jobs = []
        if "content_encoder" in stages:
            jobs.append(("content_encoder", model_dir / model["content_encoder"]["path"], {"samples": window}, shared_xdna))
        if "rmvpe" in stages and model.get("pitch"):
            jobs.append(("rmvpe", model_dir / model["pitch"]["path"], {"frames": seg // 160 + 1}, shared_xdna))
        if "generator" in stages:
            gen = next((x for x in model["generators"] if x["frames"] == g.frames and x["skip_head"] == g.skip_head
                        and x["return_length"] == g.return_length), None)
            if gen is None:
                log.error("no generator exported for %s; run convert_rvc.py --stream first", g.tag())
                return 2
            jobs.append(("generator", model_dir / gen["path"], {}, xdna_dir))
        for stage, src, dims, out_dir in jobs:
            if any(e["stage"] == stage and e["dims"] == dims and e["source"] == rel(src) for e in entries):
                continue
            static = make_static(src, dims, out_dir / "static")
            key = f"{static.stem}_{sha256_prefix(static)}"
            cache_dir = out_dir / "cache"
            log.info("[%s] %s -> %s (cache key %s)", stage, src.name, static.name, key)
            info = compile_one(stage, static, cache_dir, key, config_file, a.prepare_only)
            entries.append({"stage": stage, "dims": dims, "source": rel(src), "static_model": rel(static),
                            "cache_dir": rel(cache_dir), "cache_key": key, "config_file": rel(config_file),
                            "precision": "bf16", "onnxruntime_version": ort.__version__, **info})

    model["xdna"] = {"entries": entries, "generated_by": "tools/compile_xdna.py"}
    a.model_json.write_text(json.dumps(model, indent=2))
    compiled = sum(e["compiled"] for e in entries)
    failed = [e for e in entries if e.get("bf16_vs_fp32") and not e["bf16_vs_fp32"]["passed"]]
    log.info("wrote %d XDNA entries (%d compiled) to %s", len(entries), compiled, a.model_json)
    for e in failed:
        log.warning("[%s] BF16 output differs from FP32 beyond tolerance (rel_rms %.3f). Keep this stage on CPU/"
                    "DirectML (per-stage backend) until reviewed with tools/evaluate_conversion.py.",
                    e["stage"], e["bf16_vs_fp32"]["max_rel_rms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
