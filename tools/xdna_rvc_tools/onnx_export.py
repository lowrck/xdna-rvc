"""ONNX export, cleanup and ORT-vs-PyTorch validation helpers."""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch

from . import CONVERSION_VERSION
from .validation import Comparison, compare

log = logging.getLogger(__name__)

# Ryzen AI recommends ONNX opset 17 ("Model Compilation and Deployment").
DEFAULT_OPSET = 17


def export_onnx(
    model: torch.nn.Module,
    args: tuple,
    path: str | Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: dict | None = None,
    opset: int = DEFAULT_OPSET,
    exporter: str = "torchscript",
    metadata: dict | None = None,
) -> Path:
    """Exports `model` and stamps metadata. `exporter` is 'torchscript' (default, native
    opset 17) or 'dynamo' (torch.export based; emits opset 18 and down-converts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if exporter == "torchscript":
            torch.onnx.export(
                model, args, str(path),
                input_names=list(input_names), output_names=list(output_names),
                dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
                dynamo=False,
            )
        elif exporter == "dynamo":
            dyn_shapes = None
            if dynamic_axes:
                dyn_shapes = {}
                for name in input_names:
                    axes = dynamic_axes.get(name)
                    dyn_shapes[name] = {a: torch.export.Dim.AUTO for a in axes} if axes else None
            prog = torch.onnx.export(model, args, input_names=list(input_names), output_names=list(output_names),
                                     opset_version=opset, dynamo=True, dynamic_shapes=dyn_shapes, external_data=False)
            prog.save(str(path))
        else:
            raise ValueError(f"unknown exporter {exporter!r}")

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    props = {"xdna_rvc.conversion_version": CONVERSION_VERSION}
    for k, v in (metadata or {}).items():
        props[f"xdna_rvc.{k}"] = str(v)
    for k, v in props.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(m, str(path))
    return path


def ort_optimize_basic(src: str | Path, dst: str | Path) -> Path:
    """Constant folding + redundant node elimination with ORT's provider-independent
    'basic' level. The result contains only standard ONNX ops."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    # ORT drops metadata_props when saving; copy them back.
    src_m, dst_m = onnx.load(str(src)), onnx.load(str(dst))
    del dst_m.metadata_props[:]
    dst_m.metadata_props.extend(src_m.metadata_props)
    onnx.checker.check_model(dst_m)
    onnx.save(dst_m, str(dst))
    return Path(dst)


def run_ort(path: str | Path, feeds: dict[str, np.ndarray], provider: str = "CPUExecutionProvider") -> list[np.ndarray]:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=[provider])
    return sess.run(None, feeds)


def validate_against_torch(
    name: str,
    onnx_path: str | Path,
    torch_fn: Callable[..., torch.Tensor | tuple],
    feeds: dict[str, np.ndarray],
    max_abs_tol: float,
    rel_rms_tol: float,
    output_names: Sequence[str] | None = None,
) -> list[Comparison]:
    """Runs identical inputs through PyTorch and ONNX Runtime (CPU) and compares outputs."""
    with torch.no_grad():
        ref = torch_fn(*[torch.from_numpy(v) for v in feeds.values()])
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    t0 = time.perf_counter()
    outs = run_ort(onnx_path, feeds)
    elapsed = (time.perf_counter() - t0) * 1000
    names = list(output_names) if output_names else [f"out{i}" for i in range(len(outs))]
    comps = []
    for n, r, o in zip(names, refs, outs):
        c = compare(f"{name}:{n}", r.detach().cpu().numpy(), o, max_abs_tol, rel_rms_tol)
        comps.append(c)
        log.info("%s (ORT session+run %.0f ms)", c.line(), elapsed)
    return comps


def op_histogram(path: str | Path) -> dict[str, int]:
    m = onnx.load(str(path))
    hist: dict[str, int] = {}
    for node in m.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))
