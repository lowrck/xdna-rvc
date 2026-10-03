#!/usr/bin/env python3
"""Exact operator rewrites that keep graphs on the XDNA 2 NPU, with numerical verification.

The Ryzen AI 1.8 operator table lists no BF16 support for LeakyRelu, Relu, Softmax or
LayerNormalization (they may still be handled through fusion; the compile report decides).
If they are not supported, each occurrence becomes a CPU island and forces NPU->CPU->NPU
synchronisation. These rewrites express them with ops that *are* listed for BF16:

  LeakyRelu(x, a)     -> Max(x, Mul(x, a))                 (exact for 0 <= a <= 1)
  Relu(x)             -> Max(x, 0)                          (exact)
  Softmax(x, axis)    -> e = Exp(x - ReduceMax(x)); e / ReduceSum(e)
  LayerNorm(x, g, b)  -> d = x - mean(x); d / Sqrt(mean(d*d) + eps) * g + b

After rewriting, the model is run on the CPU against the original with random inputs and
the rewrite is rejected (exit code 1) unless outputs match to FP32 rounding. BF16 behaviour
of the rewritten graph must then be checked on hardware with compile_xdna.py, which compares
BF16 NPU output to the FP32 reference.

  python tools/optimize_xdna.py in.onnx out.onnx --rewrite all --dim samples=12000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import onnx
from onnx import helper, numpy_helper

import inspect_execution
import validate_onnx

ALL = ("leakyrelu", "relu", "softmax", "layernorm")


class Rewriter:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.g = model.graph
        self.counter = 0
        self.inits = {i.name for i in self.g.initializer}

    def name(self, base: str) -> str:
        self.counter += 1
        return f"xr_{base}_{self.counter}"

    def const(self, value: float | list, dtype=np.float32) -> str:
        n = self.name("c")
        self.g.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), n))
        return n

    def rewrite(self, kinds: set[str]) -> dict[str, int]:
        counts = {k: 0 for k in kinds}
        new_nodes = []
        for node in self.g.node:
            op = node.op_type
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            x = node.input[0] if node.input else None
            y = node.output[0] if node.output else None
            if op == "LeakyRelu" and "leakyrelu" in kinds and 0.0 <= attrs.get("alpha", 0.01) <= 1.0:
                t = self.name("lrelu_mul")
                new_nodes += [helper.make_node("Mul", [x, self.const(attrs.get("alpha", 0.01))], [t]),
                              helper.make_node("Max", [x, t], [y])]
                counts["leakyrelu"] += 1
            elif op == "Relu" and "relu" in kinds:
                new_nodes.append(helper.make_node("Max", [x, self.const(0.0)], [y]))
                counts["relu"] += 1
            elif op == "Softmax" and "softmax" in kinds:
                axis = attrs.get("axis", -1)
                ax = self.const([axis], np.int64)
                m, d, e, s = (self.name(n) for n in ("sm_max", "sm_sub", "sm_exp", "sm_sum"))
                new_nodes += [helper.make_node("ReduceMax", [x], [m], axes=[axis], keepdims=1),
                              helper.make_node("Sub", [x, m], [d]),
                              helper.make_node("Exp", [d], [e]),
                              helper.make_node("ReduceSum", [e, ax], [s], keepdims=1),
                              helper.make_node("Div", [e, s], [y])]
                counts["softmax"] += 1
            elif op == "LayerNormalization" and "layernorm" in kinds and len(node.output) == 1:
                axis = attrs.get("axis", -1)
                if axis != -1:
                    new_nodes.append(node)
                    continue
                eps = attrs.get("epsilon", 1e-5)
                mu, d, d2, var, ve, sd, n_ = (self.name(k) for k in ("ln_mu", "ln_d", "ln_d2", "ln_var", "ln_ve",
                                                                     "ln_sd", "ln_n"))
                nodes = [helper.make_node("ReduceMean", [x], [mu], axes=[-1], keepdims=1),
                         helper.make_node("Sub", [x, mu], [d]),
                         helper.make_node("Mul", [d, d], [d2]),
                         helper.make_node("ReduceMean", [d2], [var], axes=[-1], keepdims=1),
                         helper.make_node("Add", [var, self.const(eps)], [ve]),
                         helper.make_node("Sqrt", [ve], [sd]),
                         helper.make_node("Div", [d, sd], [n_])]
                scale = node.input[1]
                bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
                if bias:
                    t = self.name("ln_scaled")
                    nodes += [helper.make_node("Mul", [n_, scale], [t]), helper.make_node("Add", [t, bias], [y])]
                else:
                    nodes.append(helper.make_node("Mul", [n_, scale], [y]))
                new_nodes += nodes
                counts["layernorm"] += 1
            else:
                new_nodes.append(node)
        del self.g.node[:]
        self.g.node.extend(new_nodes)
        return counts


def opset_of(model: onnx.ModelProto) -> int:
    return max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))


def optimize(src: Path, dst: Path, kinds: set[str], dims: dict[str, int], trials: int = 3,
             rel_tol: float = 1e-4) -> dict:
    model = onnx.load(str(src))
    if opset_of(model) >= 18 and "softmax" in kinds or opset_of(model) >= 18 and "layernorm" in kinds:
        raise SystemExit("ReduceMax/ReduceMean rewrites use opset<18 attribute axes; model opset is "
                         f"{opset_of(model)} (export with opset 17)")
    before = inspect_execution.analyse_ops(src)
    counts = Rewriter(model).rewrite(kinds)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    after = inspect_execution.analyse_ops(dst)

    class A:  # minimal args object for validate_onnx.make_session
        vitis_config = vitis_cache = vitis_key = None

    ref = validate_onnx.make_session(src, "CPUExecutionProvider", dims, A)
    cand = validate_onnx.make_session(dst, "CPUExecutionProvider", dims, A)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        f = validate_onnx.feeds_for(ref, dims, rng)
        for r, c in zip(ref.run(None, f), cand.run(None, f)):
            r64, c64 = r.astype(np.float64), c.astype(np.float64)
            rel = np.sqrt(np.mean((r64 - c64) ** 2)) / max(np.sqrt(np.mean(r64 ** 2)), 1e-12)
            worst = max(worst, float(rel))
    return {"rewrites": counts, "worst_rel_rms": worst, "passed": worst <= rel_tol,
            "boundaries_before": len(before["regions"]) - 1, "boundaries_after": len(after["regions"]) - 1,
            "unsupported_after": after["unsupported_bf16"]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--rewrite", default="all", help=f"comma list of {ALL} or 'all'")
    p.add_argument("--dim", action="append", default=[], help="symbolic dim for validation inputs, name=value")
    a = p.parse_args()
    kinds = set(ALL) if a.rewrite == "all" else set(a.rewrite.split(","))
    dims = {k: int(v) for k, v in (d.split("=") for d in a.dim)}
    r = optimize(a.src, a.dst, kinds, dims)
    print(f"rewrites: {r['rewrites']}")
    print(f"predicted CPU/NPU boundaries: {r['boundaries_before']} -> {r['boundaries_after']}; "
          f"remaining unsupported: {r['unsupported_after'] or 'none'}")
    print(f"numerical check vs original (FP32 CPU): worst rel. RMS {r['worst_rel_rms']:.2e} -> "
          f"{'PASS' if r['passed'] else 'FAIL'}")
    if not r["passed"]:
        a.dst.unlink()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
