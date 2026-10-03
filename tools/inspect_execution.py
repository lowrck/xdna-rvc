#!/usr/bin/env python3
"""Where does each model (or each node) execute, and why?

Subcommands
  ops PATH.onnx [...]     Static XDNA 2 compatibility analysis: every op type in the graph is
                          checked against the Ryzen AI 1.8 supported-operator table (BF16
                          column) and the graph is split into the contiguous "NPU-candidate"
                          regions the VitisAI EP could form. Runs anywhere (no NPU needed).
                          This is a *prediction*; the compiler has the final word.
  profile FILE.json|DIR   Summarise ONNX Runtime profiles (written with --keep-profiling or by
                          `xdna-rvc-cli profile`): nodes and time per execution provider.
  report FILE.json|DIR    Summarise Vitis AI EP operator assignment reports
                          (XLNX_ONNX_EP_REPORT_FILE, written into the compile cache directory).

Examples
  python tools/inspect_execution.py ops models/shared/content_encoder_v2.onnx models/voice/generator_*.onnx
  python tools/inspect_execution.py profile diagnostics/
  python tools/inspect_execution.py report cache/vaip/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

# Ryzen AI Software 1.8 "Supported Operators", BF16 column (ryzenai.docs.amd.com/en/latest/ops_support.html,
# retrieved 2026-10-02). "Y" means broad coverage; specific configurations may still fall back.
RYZENAI_18_BF16_OPS = {
    "Abs", "Add", "And", "ArgMax", "ArgMin", "AveragePool", "BatchNormalization", "BitShift", "BitwiseAnd",
    "BitwiseNot", "BitwiseOr", "BitwiseXor", "Cast", "Ceil", "Concat", "Constant", "ConstantOfShape", "Conv",
    "ConvTranspose", "CumSum", "DepthToSpace", "Div", "Einsum", "Equal", "Erf", "Exp", "Expand", "Flatten",
    "Floor", "Gather", "GatherElements", "Gemm", "GlobalAveragePool", "Greater", "GridSample", "GroupConv",
    "Identity", "InstanceNormalization", "LSTM", "Less", "Log", "MatMul", "Max", "MaxPool", "Min", "Mod", "Mul",
    "Neg", "Not", "Or", "Pad", "Pow", "Reciprocal", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum",
    "Reshape", "Resize", "Round", "ScatterND", "Shape", "Sigmoid", "Sign", "Sin", "Size", "Slice", "Split",
    "Sqrt", "Squeeze", "Sub", "Tanh", "Tile", "Transpose", "Unsqueeze", "Upsample", "Where", "Xor",
}
# Ops absent from the BF16 column that have exact decompositions into supported ops
# (applied by tools/optimize_xdna.py; numerically verified there).
REWRITABLE = {
    "LeakyRelu": "Max(x, alpha*x)",
    "Relu": "Max(x, 0)",
    "Softmax": "Exp(x - ReduceMax) / ReduceSum",
    "LayerNormalization": "ReduceMean / Sub / Mul / Sqrt / Div",
    "Gelu": "0.5*x*(1+Erf(x/sqrt2))",
    "Clip": "Min(Max(x, lo), hi)",
}
SHAPE_ONLY = {"Shape", "Constant", "ConstantOfShape", "Identity", "Reshape", "Squeeze", "Unsqueeze", "Flatten"}


def analyse_ops(path: Path) -> dict:
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph
    hist = Counter(n.op_type for n in g.node)
    unsupported = {op: c for op, c in hist.items() if op not in RYZENAI_18_BF16_OPS}

    # Contiguous regions in topological order: how many CPU<->NPU boundaries would a
    # partitioner create if every unsupported node stayed on the CPU?
    regions = []
    for n in g.node:
        dev = "NPU" if n.op_type in RYZENAI_18_BF16_OPS else "CPU"
        if regions and regions[-1][0] == dev:
            regions[-1][1] += 1
            regions[-1][2][n.op_type] += 1
        else:
            regions.append([dev, 1, Counter({n.op_type: 1})])
    static = all(d.dim_value > 0 for i in g.input for d in i.type.tensor_type.shape.dim)
    return {
        "model": str(path),
        "opset": max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0),
        "nodes": len(g.node),
        "static_inputs": static,
        "unsupported_bf16": unsupported,
        "rewritable": {op: REWRITABLE[op] for op in unsupported if op in REWRITABLE},
        "blocking": {op: c for op, c in unsupported.items() if op not in REWRITABLE},
        "npu_candidate_nodes": sum(c for op, c in hist.items() if op in RYZENAI_18_BF16_OPS),
        "regions": [{"device": d, "nodes": n, "ops": dict(c.most_common(4))} for d, n, c in regions],
    }


def cmd_ops(paths: list[Path], as_json: bool) -> int:
    results = [analyse_ops(p) for p in paths]
    if as_json:
        print(json.dumps(results, indent=2))
        return 0
    for r in results:
        cpu_regions = [x for x in r["regions"] if x["device"] == "CPU"]
        print(f"\n{r['model']}")
        print(f"  opset {r['opset']}, {r['nodes']} nodes, static input shapes: {r['static_inputs']}")
        print(f"  ops in the Ryzen AI 1.8 BF16 table: {r['npu_candidate_nodes']}/{r['nodes']} nodes")
        if r["rewritable"]:
            print("  not in BF16 table, exact rewrite available:")
            for op, how in r["rewritable"].items():
                print(f"    {op:20s} x{r['unsupported_bf16'][op]:<4d} -> {how}")
        if r["blocking"]:
            print("  not in BF16 table, no rewrite (stays on CPU):")
            for op, c in r["blocking"].items():
                print(f"    {op:20s} x{c}")
        print(f"  CPU/NPU boundaries if unsupported ops stay on CPU: {len(r['regions']) - 1} "
              f"({len(cpu_regions)} CPU islands)")
        if not r["static_inputs"]:
            print("  NOTE: dynamic input shapes - the runtime pins them (free dimension overrides) before XDNA compile")
    return 0


def _files(path: Path, pattern: str) -> list[Path]:
    return sorted(path.rglob(pattern)) if path.is_dir() else [path]


def cmd_profile(path: Path, as_json: bool) -> int:
    out = []
    for f in _files(path, "ort_profile_*.json"):
        events = json.loads(f.read_text())
        per = defaultdict(lambda: {"nodes": set(), "us": 0.0, "ops": Counter()})
        for ev in events:
            if ev.get("cat") != "Node" or not ev.get("name", "").endswith("_kernel_time"):
                continue
            args = ev.get("args", {})
            p = per[args.get("provider", "?")]
            p["nodes"].add(ev["name"])
            p["us"] += ev.get("dur", 0)
            p["ops"][args.get("op_name", "?")] += 1
        total = sum(p["us"] for p in per.values()) or 1.0
        summary = {prov: {"nodes": len(p["nodes"]), "time_pct": 100 * p["us"] / total,
                          "top_ops": dict(p["ops"].most_common(5))} for prov, p in per.items()}
        out.append({"file": str(f), "providers": summary})
    if as_json:
        print(json.dumps(out, indent=2))
    else:
        for o in out:
            print(f"\n{o['file']}")
            for prov, s in sorted(o["providers"].items(), key=lambda kv: -kv[1]["time_pct"]):
                print(f"  {prov:28s} {s['nodes']:5d} nodes {s['time_pct']:6.1f}% time  {s['top_ops']}")
    if not out:
        print(f"no ort_profile_*.json under {path}")
    return 0


def cmd_report(path: Path, as_json: bool) -> int:
    reports = []
    for f in _files(path, "vitisai_ep_report.json"):
        j = json.loads(f.read_text())
        stats = {s.get("name"): {"nodes": s.get("nodeNum", 0), "ops": s.get("supportedOpType", [])}
                 for s in j.get("deviceStat", [])}
        reports.append({"file": str(f), "device_stats": stats})
    if as_json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            st = r["device_stats"]
            total = st.get("all", {}).get("nodes", 0)
            print(f"\n{r['file']}: {total} nodes")
            for dev, s in st.items():
                if dev == "all":
                    continue
                pct = 100 * s["nodes"] / total if total else 0
                print(f"  {dev:6s} {s['nodes']:5d} nodes ({pct:5.1f}%)  ops: {', '.join(s['ops'][:12])}")
    if not reports:
        print(f"no vitisai_ep_report.json under {path} (set enable_cache_file_io_in_mem=0 and "
              "XLNX_ONNX_EP_REPORT_FILE; xdna-rvc does both automatically)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("ops")
    o.add_argument("models", nargs="+", type=Path)
    pr = sub.add_parser("profile")
    pr.add_argument("path", type=Path)
    rp = sub.add_parser("report")
    rp.add_argument("path", type=Path)
    a = p.parse_args()
    if a.cmd == "ops":
        return cmd_ops(a.models, a.json)
    if a.cmd == "profile":
        return cmd_profile(a.path, a.json)
    return cmd_report(a.path, a.json)


if __name__ == "__main__":
    sys.exit(main())
