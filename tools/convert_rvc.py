#!/usr/bin/env python3
"""Convert an RVC voice model (.pth, optional .index) into an xdna-rvc model directory.

Workflow
  1. Load the checkpoint safely (weights_only) and identify the architecture from
     its weights: RVC v1/v2, F0 or not, sample rate, speakers, decoder layout.
  2. Rebuild the upstream RVC synthesizer and load the weights strictly.
  3. For each streaming geometry: build the fixed-shape export model, check it
     against upstream RVC infer() (same weights, same injected noise), export ONNX
     (opset 17), clean the graph with ORT basic optimisation, and validate ONNX
     Runtime against PyTorch.
  4. Export the FAISS .index into arrays for the native IVF search and verify the
     native algorithm reproduces FAISS results.
  5. Write model.json (+ conversion_report.json).

Re-running with an unchanged .pth reuses existing outputs (hash + conversion version)
unless --force is given.

Examples
  python tools/convert_rvc.py voice.pth --index added_IVF256_Flat_nprobe_1_voice_v2.index
  python tools/convert_rvc.py G_2333.pth --sample-rate 40000 --preset low_latency --stream 80,40,500
  python tools/convert_rvc.py voice.pth --inspect-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from xdna_rvc_tools import CONVERSION_VERSION
from xdna_rvc_tools.onnx_export import export_onnx, op_histogram, ort_optimize_basic, validate_against_torch
from xdna_rvc_tools.rvc.checkpoint import UnsupportedModelError, build_upstream_synthesizer, inspect_checkpoint
from xdna_rvc_tools.rvc.export_model import GeneratorExport, example_inputs, reference_infer
from xdna_rvc_tools.safe_load import UnsafeCheckpointError, sha256_file
from xdna_rvc_tools.stream_config import PRESETS, StreamConfig, parse_stream
from xdna_rvc_tools.validation import Comparison, compare, write_report

log = logging.getLogger("convert_rvc")

# Export model vs upstream RVC (same math, different op order): tight.
REF_MAX_ABS, REF_REL_RMS = 1e-4, 1e-4
# ONNX Runtime vs PyTorch, FP32.
ORT_MAX_ABS, ORT_REL_RMS = 1e-3, 1e-4


def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return name or "voice"


def load_existing(model_dir: Path) -> dict | None:
    p = model_dir / "model.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def export_generator(info, sd, upstream, cfg: StreamConfig, out_dir: Path, exporter: str) -> tuple[dict, list[Comparison]]:
    geom = cfg.geometry()
    t0 = time.perf_counter()
    model = GeneratorExport(info, sd, geom).eval()
    comps: list[Comparison] = []

    # 1. export model == upstream RVC infer()
    for seed in (0, 1):
        x = example_inputs(model, seed=seed)
        with torch.no_grad():
            ours = model(*x.values())
        ref = reference_infer(upstream, info, geom, x)
        c = compare(f"{geom.tag()}/export_vs_upstream_seed{seed}", ref.numpy(), ours.numpy(), REF_MAX_ABS, REF_REL_RMS)
        log.info("%s", c.line())
        comps.append(c)

    # 2. ONNX export + ORT vs PyTorch
    name = f"generator_{geom.tag()}.onnx"
    raw = out_dir / (name + ".raw")
    final = out_dir / name
    x = example_inputs(model, seed=2)
    export_onnx(model, tuple(x.values()), raw, list(x.keys()), ["audio"], exporter=exporter,
                metadata={"stage": "generator", "rvc_version": info.version, "uses_f0": int(info.uses_f0),
                          "sample_rate": info.sample_rate, "frames": geom.frames, "skip_head": geom.skip_head,
                          "return_length": geom.return_length, "source_sha256": info.sha256})
    ort_optimize_basic(raw, final)
    raw.unlink()
    for seed in (3, 4):
        feeds = {k: v.numpy() for k, v in example_inputs(model, seed=seed).items()}
        comps += validate_against_torch(f"{geom.tag()}/onnx_seed{seed}", final, model, feeds, ORT_MAX_ABS, ORT_REL_RMS,
                                        ["audio"])
    entry = {
        "path": name,
        "stream": {"block_ms": cfg.block_ms, "crossfade_ms": cfg.crossfade_ms, "extra_ms": cfg.extra_ms,
                   "lookahead_ms": cfg.lookahead_ms},
        "frames": geom.frames,
        "skip_head": geom.skip_head,
        "return_length": geom.return_length,
        "flow_head": geom.flow_head,
        "output_samples": geom.return_length * info.upsample_factor,
        "inputs": {k: list(s) for k, (s, _) in model.input_shapes().items()},
        "precision": "fp32",
        "sha256": sha256_file(final),
        "validation": {c.name: {"max_abs": c.max_abs, "rel_rms": c.rel_rms, "passed": c.passed} for c in comps},
    }
    log.info("exported %s in %.1f s (ops: %s)", final.name, time.perf_counter() - t0,
             ", ".join(f"{k}:{v}" for k, v in list(op_histogram(final).items())[:8]))
    return entry, comps


def convert_index(index_path: Path, out_dir: Path, feature_dim: int) -> tuple[dict, list[Comparison]]:
    from xdna_rvc_tools.rvc.index import ivf_search, read_faiss_index, write_index_dir
    import faiss

    arrays = read_faiss_index(index_path)
    if arrays.dim != feature_dim:
        raise UnsupportedModelError(f"{index_path.name}: index dim {arrays.dim} does not match the model's "
                                    f"{feature_dim}-dim features (v1 models need 256-dim, v2 768-dim indexes)")
    write_index_dir(arrays, out_dir, index_path)
    # Verify the native algorithm (numpy reference of the C++ search) against FAISS.
    idx = faiss.read_index(str(index_path))
    rng = np.random.default_rng(0)
    pick = arrays.vectors[rng.choice(len(arrays.vectors), min(128, len(arrays.vectors)), replace=False)]
    q = (pick + rng.standard_normal(pick.shape).astype(np.float32) * 0.05).astype(np.float32)
    D, I = idx.search(q, 8)
    d, rows = ivf_search(arrays, q, 8)
    ids = np.where(rows >= 0, arrays.ids[np.maximum(rows, 0)], -1)
    agree = float(np.mean(ids == I))
    c = compare("index/ivf_distances", D, d, 1e-2, 1e-4)
    c.passed = c.passed and agree >= 0.99
    c.notes.append(f"neighbour id agreement {agree:.4f}")
    log.info("%s; %s", c.line(), c.notes[-1])
    meta = json.loads((out_dir / "index.json").read_text())
    meta["path"] = out_dir.name
    return meta, [c]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pth", type=Path, help="RVC voice checkpoint (.pth)")
    p.add_argument("--index", type=Path, help="RVC FAISS index (added_*.index)")
    p.add_argument("--name", help="model directory name (default: .pth file stem)")
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--shared-dir", type=Path, help="shared models dir (default: <models-dir>/shared)")
    p.add_argument("--preset", action="append", choices=list(PRESETS), help="streaming preset(s) to export")
    p.add_argument("--stream", action="append", default=[],
                   help="extra geometry block_ms,crossfade_ms,extra_ms[,lookahead_ms]")
    p.add_argument("--sample-rate", type=int, choices=[32000, 40000, 48000],
                   help="sample rate for training checkpoints without metadata")
    p.add_argument("--exporter", choices=["torchscript", "dynamo"], default="torchscript")
    p.add_argument("--force", action="store_true", help="re-export even if cached outputs match")
    p.add_argument("--inspect-only", action="store_true", help="print the checkpoint analysis and exit")
    p.add_argument("--allow-unsafe-pickle", action="store_true",
                   help="load checkpoints that need full pickle deserialization (only for files you trust)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    _bootstrap.setup_logging(args.verbose)

    try:
        info, sd = inspect_checkpoint(args.pth, sr_hint=args.sample_rate, allow_unsafe=args.allow_unsafe_pickle)
    except (UnsupportedModelError, UnsafeCheckpointError, FileNotFoundError) as e:
        log.error("%s", e)
        return 2
    log.info("checkpoint %s: %s layout, RVC %s (%d-dim features), %s, %d Hz, %d speaker(s), %s",
             args.pth.name, info.layout, info.version, info.feature_dim, "F0" if info.uses_f0 else "no F0",
             info.sample_rate, info.n_speakers, info.synthesizer)
    for w in info.warnings:
        log.warning("%s", w)
    if args.inspect_only:
        print(json.dumps(info.to_json(), indent=2))
        return 0

    name = safe_name(args.name or args.pth.stem)
    model_dir = args.models_dir / name
    shared = args.shared_dir or (args.models_dir / "shared")
    model_dir.mkdir(parents=True, exist_ok=True)

    configs = [PRESETS[x] for x in (args.preset or ["balanced", "low_latency"])] + [parse_stream(s) for s in args.stream]
    configs = list(dict.fromkeys(configs))

    existing = load_existing(model_dir)
    reuse: dict[str, dict] = {}
    if existing and not args.force and existing.get("source", {}).get("sha256") == info.sha256 \
            and existing.get("conversion_version") == CONVERSION_VERSION:
        for g in existing.get("generators", []):
            if (model_dir / g["path"]).is_file():
                reuse[g["path"]] = g
        log.info("model.json matches this checkpoint; reusing %d existing generator export(s)", len(reuse))

    upstream = build_upstream_synthesizer(info, sd)
    generators, comps = [], []
    for cfg in configs:
        fname = f"generator_{cfg.geometry().tag()}.onnx"
        if fname in reuse:
            log.info("reusing %s", fname)
            generators.append(reuse[fname])
            continue
        entry, c = export_generator(info, sd, upstream, cfg, model_dir, args.exporter)
        generators.append(entry)
        comps += c

    index_meta = None
    if args.index:
        index_meta, c = convert_index(args.index, model_dir / "index", info.feature_dim)
        comps += c
    elif existing and existing.get("index") and not args.force and (model_dir / "index").is_dir():
        index_meta = existing["index"]

    def rel(p: Path) -> str:
        return os.path.relpath(p, model_dir).replace(os.sep, "/")

    content = shared / f"content_encoder_{info.version}.onnx"
    rmvpe = shared / "rmvpe.onnx"
    for path, tool in ((content, "convert_contentvec.py"), (rmvpe, "convert_rmvpe.py")):
        if not path.is_file() and (path != rmvpe or info.uses_f0):
            log.warning("shared model %s not found yet; create it with tools/%s", path, tool)

    model_json = {
        "schema": 1,
        "name": name,
        "conversion_version": CONVERSION_VERSION,
        "source": {"file": args.pth.name, "sha256": info.sha256, "layout": info.layout,
                   "index_file": args.index.name if args.index else (existing or {}).get("source", {}).get("index_file")},
        "rvc": {
            "version": info.version, "feature_dim": info.feature_dim, "uses_f0": info.uses_f0,
            "sample_rate": info.sample_rate, "n_speakers": info.n_speakers,
            "speaker_names": {str(k): v for k, v in info.speaker_names.items()},
            "upsample_factor": info.upsample_factor, "synthesizer": info.synthesizer,
            "inter_channels": info.config["inter_channels"], "info": info.info,
        },
        "content_encoder": {"path": rel(content)},
        "pitch": {"method": "rmvpe", "path": rel(rmvpe)} if info.uses_f0 else None,
        "generators": generators,
        "index": index_meta,
        "warnings": info.warnings,
    }
    # Keep XDNA precompile entries (tools/compile_xdna.py) that still refer to current files.
    if existing and existing.get("xdna"):
        gen_paths = {g["path"] for g in generators}
        kept = [e for e in existing["xdna"].get("entries", [])
                if e.get("stage") != "generator" or e.get("source") in gen_paths]
        if existing.get("source", {}).get("sha256") != info.sha256:
            kept = [e for e in kept if e.get("stage") != "generator"]  # weights changed: recompile generators
        if kept:
            model_json["xdna"] = {**existing["xdna"], "entries": kept}
    (model_dir / "model.json").write_text(json.dumps(model_json, indent=2))
    write_report(model_dir / "conversion_report.json", comps, {"checkpoint": info.to_json()})
    ok = all(c.passed for c in comps)
    if ok:
        log.info("wrote %s (%d generator variant(s)%s)", model_dir / "model.json", len(generators),
                 ", index" if index_meta else "")
    else:
        failed = [c.name for c in comps if not c.passed]
        log.error("VALIDATION FAILED for %s; see %s. The exported model must not be trusted.", failed,
                  model_dir / "conversion_report.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
