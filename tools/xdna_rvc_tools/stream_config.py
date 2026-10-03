"""Streaming configuration -> fixed model geometry.

This mirrors src/rvc/stream_config.{h,cpp}; both are tested against the same table.

Time unit: 10 ms frames (RVC's f0/feature frame rate after 2x upsampling of HuBERT).

    H  = block_ms / 10                new audio per inference (the hop)
    C  = crossfade_ms / 10            crossfade length
    Cb = min(C, 4)                    SOLA buffer (max 40 ms, as upstream)
    S  = 1                            SOLA search window (10 ms)
    E  = extra_ms / 10                left context for the neural models
    L  = lookahead_ms / 10            right context (future audio) beyond the decoded region

    window frames T = E + C + S + H + L       (generator/content-encoder input length)
    skip_head       = E                       (context frames not decoded)
    return_length   = H + Cb + S              (frames decoded per inference)

With L = 0 this is the structure of upstream RVC's realtime_gui.py. L > 0 is an xdna-rvc
addition: the decoded frames otherwise sit at the very end of the window, where the
bidirectional HuBERT/attention layers have no future context; L buys quality for latency.
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
    lookahead_ms: int = 0

    def validate(self) -> None:
        for name in ("block_ms", "crossfade_ms", "extra_ms", "lookahead_ms"):
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
        la = self.lookahead_ms // FRAME_MS
        cb = min(c, SOLA_BUFFER_MAX_FRAMES)
        return StreamGeometry(frames=e + c + SOLA_SEARCH_FRAMES + h + la, skip_head=e,
                              return_length=h + cb + SOLA_SEARCH_FRAMES)


# Chosen from measurements (tools/eval_stream_quality.py, 20 LibriSpeech utterances, ASR WER;
# see docs/latency.md): 60 ms of lookahead lets a 40-60 ms hop match the intelligibility of
# upstream's 100 ms hop / 600 ms context configuration.
PRESETS = {
    "low_latency": StreamConfig(block_ms=40, crossfade_ms=20, extra_ms=300, lookahead_ms=60),
    "balanced": StreamConfig(block_ms=60, crossfade_ms=20, extra_ms=300, lookahead_ms=60),
    "quality": StreamConfig(block_ms=100, crossfade_ms=40, extra_ms=600, lookahead_ms=0),
}


def parse_stream(text: str) -> StreamConfig:
    """'block,crossfade,extra[,lookahead]' in ms, e.g. '100,40,600' or '40,20,300,40', or a preset."""
    if text in PRESETS:
        return PRESETS[text]
    parts = [int(x) for x in text.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError(f"stream spec '{text}': expected block_ms,crossfade_ms,extra_ms[,lookahead_ms] or one of "
                         f"{list(PRESETS)}")
    cfg = StreamConfig(*parts)
    cfg.validate()
    return cfg
