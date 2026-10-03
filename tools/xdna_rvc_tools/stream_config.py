"""Streaming configuration -> fixed model geometry.

This mirrors src/rvc/stream_config.{h,cpp}; both are tested against the same table.

Time unit: 10 ms frames (RVC's f0/feature frame rate after 2x upsampling of HuBERT).

    H  = block_ms / 10                new audio per inference (the hop)
    C  = crossfade_ms / 10            crossfade length
    Cb = min(C, 4)                    SOLA buffer (max 40 ms, as upstream)
    S  = 1                            SOLA search window (10 ms)
    E  = extra_ms / 10                left context for the neural models

    window frames T = E + C + S + H           (generator/content-encoder input length)
    skip_head       = E                       (context frames not decoded)
    return_length   = H + Cb + S              (frames decoded per inference)

Same structure as upstream RVC's realtime_gui.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rvc.export_model import StreamGeometry

FRAME_MS = 10
SOLA_SEARCH_FRAMES = 1
SOLA_BUFFER_MAX_FRAMES = 4


@dataclass(frozen=True)
class StreamConfig:
    block_ms: int = 100
    crossfade_ms: int = 40
    extra_ms: int = 600

    def validate(self) -> None:
        for name in ("block_ms", "crossfade_ms", "extra_ms"):
            v = getattr(self, name)
            if v < 0 or v % FRAME_MS:
                raise ValueError(f"{name}={v} must be a non-negative multiple of {FRAME_MS} ms")
        if self.block_ms < 2 * FRAME_MS:
            raise ValueError("block_ms must be at least 20 ms")
        if self.crossfade_ms < FRAME_MS:
            raise ValueError("crossfade_ms must be at least 10 ms")

    def geometry(self) -> StreamGeometry:
        self.validate()
        h = self.block_ms // FRAME_MS
        c = self.crossfade_ms // FRAME_MS
        e = self.extra_ms // FRAME_MS
        cb = min(c, SOLA_BUFFER_MAX_FRAMES)
        return StreamGeometry(frames=e + c + SOLA_SEARCH_FRAMES + h, skip_head=e,
                              return_length=h + cb + SOLA_SEARCH_FRAMES)


PRESETS = {
    "low_latency": StreamConfig(block_ms=60, crossfade_ms=30, extra_ms=400),
    "balanced": StreamConfig(block_ms=100, crossfade_ms=40, extra_ms=600),
    "quality": StreamConfig(block_ms=200, crossfade_ms=60, extra_ms=1000),
}


def parse_stream(text: str) -> StreamConfig:
    """'block,crossfade,extra' in ms, e.g. '100,40,600', or a preset name."""
    if text in PRESETS:
        return PRESETS[text]
    parts = [int(x) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"stream spec '{text}': expected block_ms,crossfade_ms,extra_ms or one of {list(PRESETS)}")
    cfg = StreamConfig(*parts)
    cfg.validate()
    return cfg
