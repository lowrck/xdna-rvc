"""Shared fixtures for the Python test-suite.

Tests that need downloaded model weights or a built C++ binary skip themselves
when those are absent. Locations can be overridden with environment variables:
  XDNA_RVC_TEST_ASSETS  directory with hubert_base/, rmvpe/, speech/, voices/ (default: test_assets)
  XDNA_RVC_CLI          path to xdna-rvc-cli (default: build/src/xdna-rvc-cli[.exe])
  XDNA_RVC_MODELS       converted shared models dir (default: models/shared)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))


def _path_from_env(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default)))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def assets() -> Path:
    return _path_from_env("XDNA_RVC_TEST_ASSETS", ROOT / "test_assets")


@pytest.fixture(scope="session")
def hubert_dir(assets: Path) -> Path:
    p = assets / "hubert_base"
    if not (p / "pytorch_model.bin").is_file():
        pytest.skip(f"HuBERT weights not found in {p} (see docs/model_conversion.md)")
    return p


@pytest.fixture(scope="session")
def rmvpe_path(assets: Path) -> Path:
    p = assets / "rmvpe" / "rmvpe.pt"
    if not p.is_file():
        pytest.skip(f"RMVPE weights not found at {p}")
    return p


@pytest.fixture(scope="session")
def speech_wav(assets: Path) -> Path:
    p = assets / "speech" / "utt0_48k.wav"
    if not p.is_file():
        pytest.skip(f"test speech not found at {p}")
    return p


@pytest.fixture(scope="session")
def cli() -> Path:
    exe = "xdna-rvc-cli.exe" if os.name == "nt" else "xdna-rvc-cli"
    candidates = [ROOT / "build" / "src" / exe, ROOT / "build" / "src" / "Release" / exe]
    env = os.environ.get("XDNA_RVC_CLI")
    if env:
        candidates.insert(0, Path(env))
    for c in candidates:
        if c.is_file():
            return c
    pytest.skip("xdna-rvc-cli not built")


@pytest.fixture(scope="session")
def shared_models() -> Path:
    p = _path_from_env("XDNA_RVC_MODELS", ROOT / "models" / "shared")
    if not p.is_dir():
        pytest.skip(f"converted shared models not found in {p} (run tools/convert_contentvec.py)")
    return p
