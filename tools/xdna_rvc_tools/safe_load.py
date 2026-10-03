"""Loading untrusted PyTorch checkpoints without executing pickled code.

Voice models are downloaded from the internet and must be treated as untrusted.
`torch.load(..., weights_only=True)` restricts unpickling to tensors and plain
containers, so a malicious checkpoint cannot run code during loading. We never
fall back to unrestricted unpickling automatically: if a checkpoint needs it, the
user must opt in explicitly (`allow_unsafe=True`) after being warned.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded with weights_only=True."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path: str | Path, allow_unsafe: bool = False) -> Any:
    """Loads a .pt/.pth/.bin checkpoint safely.

    Raises UnsafeCheckpointError (with the restricted-unpickler message) if the file
    contains objects other than tensors/containers and allow_unsafe is False.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises UnpicklingError subclasses with details
        if not allow_unsafe:
            raise UnsafeCheckpointError(
                f"{path.name} cannot be loaded with weights_only=True (safe mode): {exc}\n"
                "The file contains pickled Python objects beyond tensors and plain containers. "
                "Loading it would execute code from the file. Only proceed if you fully trust its "
                "source, by passing --allow-unsafe-pickle."
            ) from exc
        log.warning(
            "UNSAFE: loading %s with full pickle deserialization because --allow-unsafe-pickle was given. "
            "This executes code embedded in the file.",
            path,
        )
        return torch.load(path, map_location="cpu", weights_only=False)
