"""XDNA tooling that can be verified without an NPU, plus hardware tests that skip
themselves unless the VitisAI execution provider is available."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import pytest

VITIS = "VitisAIExecutionProvider"


def test_static_model_loads_at_full_optimisation_and_matches(shared_models, tmp_path):
    import compile_xdna
    src = shared_models / "content_encoder_v2.onnx"
    if not src.is_file():
        pytest.skip("content encoder not converted")
    static = compile_xdna.make_static(src, {"samples": 8000}, tmp_path)
    m = __import__("onnx").load(str(static), load_external_data=False)
    assert [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim] == [1, 8000]
    x = (np.random.default_rng(0).standard_normal((1, 8000)) * 0.1).astype(np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # regression: ORT 1.30 bug
    y = ort.InferenceSession(str(static), so, providers=["CPUExecutionProvider"]).run(None, {"audio": x})[0]
    ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, {"audio": x})[0]
    np.testing.assert_allclose(y, ref, atol=1e-4)


def test_op_analysis_flags_gru_as_cpu_only(shared_models):
    import inspect_execution
    p = shared_models / "rmvpe.onnx"
    if not p.is_file():
        pytest.skip("rmvpe not converted")
    r = inspect_execution.analyse_ops(p)
    assert r["blocking"] == {"GRU": 1}
    assert "Relu" in r["rewritable"]


def test_prepare_only_writes_manifest(root, tmp_path):
    model = root / "models" / "vctk_v2_f0_40k" / "model.json"
    if not model.is_file():
        pytest.skip("test voice not converted")
    j = json.loads(model.read_text())
    entries = j.get("xdna", {}).get("entries", [])
    if not entries:
        pytest.skip("run tools/compile_xdna.py --prepare-only first")
    stages = {e["stage"] for e in entries}
    assert {"content_encoder", "rmvpe", "generator"} <= stages
    for e in entries:
        assert e["precision"] == "bf16"
        assert (model.parent / e["static_model"]).is_file()


@pytest.mark.hardware
@pytest.mark.skipif(VITIS not in ort.get_available_providers(), reason="VitisAI EP not available (needs Ryzen AI on STX/KRK)")
def test_compile_and_bf16_accuracy_on_npu(root):
    model = root / "models" / "vctk_v2_f0_40k" / "model.json"
    r = subprocess.run([sys.executable, str(root / "tools" / "compile_xdna.py"), str(model), "--preset", "balanced"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    entries = json.loads(model.read_text())["xdna"]["entries"]
    for e in entries:
        assert e["compiled"]
        assert e["assignment"].get("NPU", 0) > 0, f"{e['stage']}: no nodes on the NPU: {e['assignment']}"


def test_rewrites_are_exact_and_remove_unsupported_ops(tmp_path):
    """Random-weight generator export: all four rewrite kinds apply and match FP32."""
    import torch
    import optimize_xdna
    from xdna_rvc_tools.onnx_export import export_onnx, ort_optimize_basic
    from xdna_rvc_tools.rvc.checkpoint import inspect_checkpoint
    from xdna_rvc_tools.rvc.export_model import GeneratorExport, StreamGeometry, example_inputs
    from test_rvc_conversion import random_checkpoint

    info, sd = inspect_checkpoint(random_checkpoint(tmp_path, "v2", True, 40000))
    model = GeneratorExport(info, sd, StreamGeometry(40, 10, 10)).eval()
    x = example_inputs(model, seed=0)
    raw, src, dst = tmp_path / "r.onnx", tmp_path / "g.onnx", tmp_path / "g_rw.onnx"
    export_onnx(model, tuple(x.values()), raw, list(x.keys()), ["audio"])
    ort_optimize_basic(raw, src)
    r = optimize_xdna.optimize(src, dst, set(optimize_xdna.ALL), {})
    assert r["passed"], r
    assert r["rewrites"]["leakyrelu"] > 0 and r["rewrites"]["layernorm"] > 0 and r["rewrites"]["softmax"] > 0
    assert r["unsupported_after"] == {}
    assert r["boundaries_after"] == 0
