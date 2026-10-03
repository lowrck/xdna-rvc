#!/usr/bin/env python3
"""Benchmark a single ONNX model on one or more execution providers.

Reports session creation time (includes NPU compilation, or cache load when a compiled
cache exists) and run latency statistics, as text and optionally JSON. Use
`xdna-rvc-cli benchmark` for the whole streaming pipeline; this tool isolates one stage.

  python tools/benchmark_model.py models/shared/xdna/static/content_encoder_v2__samples12000.onnx \
      --provider CPUExecutionProvider --provider VitisAIExecutionProvider \
      --vitis-config models/voice/xdna/vaiml_config.json --vitis-cache models/shared/xdna/cache \
      --vitis-key <key from model.json> --json-out bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

import validate_onnx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path)
    p.add_argument("--provider", action="append", help="execution provider(s), default CPU")
    p.add_argument("--dim", action="append", default=[], help="symbolic dim override name=value")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--threads", type=int, default=0, help="CPU intra-op threads (0 = ORT default)")
    p.add_argument("--vitis-config")
    p.add_argument("--vitis-cache")
    p.add_argument("--vitis-key")
    p.add_argument("--json-out", type=Path)
    a = p.parse_args()
    dims = {k: int(v) for k, v in (d.split("=") for d in a.dim)}
    results = []
    for prov in a.provider or ["CPUExecutionProvider"]:
        t0 = time.perf_counter()
        sess = validate_onnx.make_session(a.model, prov, dims, a)
        create_ms = (time.perf_counter() - t0) * 1000
        feeds = validate_onnx.feeds_for(sess, dims, np.random.default_rng(0))
        for _ in range(a.warmup):
            sess.run(None, feeds)
        times = []
        for _ in range(a.runs):
            t = time.perf_counter()
            sess.run(None, feeds)
            times.append((time.perf_counter() - t) * 1000)
        t = np.array(times)
        r = {"model": str(a.model), "provider": prov, "session_create_ms": round(create_ms, 1),
             "mean_ms": float(t.mean()), "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
             "max_ms": float(t.max()), "runs": a.runs}
        results.append(r)
        print(f"{prov:28s} create {create_ms:9.1f} ms | run mean {r['mean_ms']:7.2f}  median {r['median_ms']:7.2f}"
              f"  p95 {r['p95_ms']:7.2f}  max {r['max_ms']:7.2f} ms")
    if a.json_out:
        a.json_out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
