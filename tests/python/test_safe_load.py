"""Untrusted checkpoints must not execute code when loaded."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xdna_rvc_tools.safe_load import UnsafeCheckpointError, load_checkpoint, sha256_file

MARKER: Path | None = None


def _touch_marker(path: str) -> None:
    Path(path).write_text("code was executed during unpickling")


class Malicious:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (_touch_marker, (str(self.marker),))


def test_plain_state_dict_loads(tmp_path: Path) -> None:
    p = tmp_path / "ok.pth"
    torch.save({"weight": {"a": torch.ones(3)}, "config": [1, 2, "40k"], "version": "v2", "f0": 1}, p)
    ck = load_checkpoint(p)
    assert ck["version"] == "v2"
    assert torch.equal(ck["weight"]["a"], torch.ones(3))


def test_malicious_pickle_is_rejected_without_executing(tmp_path: Path) -> None:
    marker = tmp_path / "pwned.txt"
    p = tmp_path / "evil.pth"
    torch.save({"weight": {}, "payload": Malicious(marker)}, p)
    assert not marker.exists()
    with pytest.raises(UnsafeCheckpointError) as info:
        load_checkpoint(p)
    assert "weights_only" in str(info.value)
    assert not marker.exists(), "safe loader executed pickled code"


def test_sha256_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
