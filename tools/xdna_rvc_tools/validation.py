"""Numerical comparison between a reference (PyTorch) and a candidate (ONNX Runtime) output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Comparison:
    name: str
    shape_ref: tuple
    shape_out: tuple
    max_abs: float
    mean_abs: float
    rms_ref: float
    rel_rms: float          # rms(diff) / rms(ref)
    cosine: float           # cosine similarity of flattened tensors
    tolerance_max_abs: float
    tolerance_rel_rms: float
    passed: bool
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: max|d|={self.max_abs:.3e} mean|d|={self.mean_abs:.3e} "
            f"rel_rms={self.rel_rms:.3e} cos={self.cosine:.6f} "
            f"(tol max|d|<={self.tolerance_max_abs:g}, rel_rms<={self.tolerance_rel_rms:g})"
        )


def compare(
    name: str,
    ref: np.ndarray,
    out: np.ndarray,
    max_abs_tol: float,
    rel_rms_tol: float,
) -> Comparison:
    ref = np.asarray(ref, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    notes: list[str] = []
    if ref.shape != out.shape:
        return Comparison(name, ref.shape, out.shape, float("inf"), float("inf"), 0.0, float("inf"), 0.0,
                          max_abs_tol, rel_rms_tol, False, [f"shape mismatch {ref.shape} vs {out.shape}"])
    if not np.all(np.isfinite(out)):
        notes.append("candidate contains NaN/Inf")
    diff = out - ref
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
    rms_ref = float(np.sqrt(np.mean(ref * ref))) if ref.size else 0.0
    rms_diff = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    rel_rms = rms_diff / rms_ref if rms_ref > 0 else (0.0 if rms_diff == 0 else float("inf"))
    denom = float(np.linalg.norm(ref) * np.linalg.norm(out))
    cosine = float(np.dot(ref.ravel(), out.ravel()) / denom) if denom > 0 else 1.0
    passed = (not notes) and max_abs <= max_abs_tol and rel_rms <= rel_rms_tol
    return Comparison(name, ref.shape, out.shape, max_abs, mean_abs, rms_ref, rel_rms, cosine,
                      max_abs_tol, rel_rms_tol, passed, notes)


def write_report(path: str | Path, comparisons: list[Comparison], extra: dict | None = None) -> None:
    data = {"comparisons": [asdict(c) for c in comparisons], "all_passed": all(c.passed for c in comparisons)}
    if extra:
        data.update(extra)
    Path(path).write_text(json.dumps(data, indent=2, default=str))
