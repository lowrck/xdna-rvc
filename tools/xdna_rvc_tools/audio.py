"""Audio file helpers for the conversion/validation tools."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Returns (mono float32 audio, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), int(sr)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return audio.astype(np.float32)
    g = gcd(sr_from, sr_to)
    return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


def load_wav_mono_16k(path: str | Path) -> np.ndarray:
    audio, sr = load_wav(path)
    return resample(audio, sr, 16000)
