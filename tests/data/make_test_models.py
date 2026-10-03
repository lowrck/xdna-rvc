"""Generates the tiny ONNX models used by the C++ unit tests.

Run from the repository root:  python tests/data/make_test_models.py
The outputs are small and committed so C++ tests do not need Python.
"""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent


def make_add_mul() -> None:
    # y = (x + 1) * 2, x: float32[1, N] with dynamic N
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "n"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "n"])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    nodes = [
        helper.make_node("Add", ["x", "one"], ["t"], name="add"),
        helper.make_node("Mul", ["t", "two"], ["y"], name="mul"),
    ]
    graph = helper.make_graph(nodes, "add_mul", [x], [y], initializer=[one, two])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="xdna-rvc-tests")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, HERE / "add_mul.onnx")


if __name__ == "__main__":
    make_add_mul()
    print("wrote", HERE / "add_mul.onnx")
