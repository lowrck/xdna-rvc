#pragma once

#include <string>

namespace xr::rvc {

// Streaming configuration in milliseconds. Mirrors tools/xdna_rvc_tools/stream_config.py
// (both are tested against the same table). Time unit internally: 10 ms frames.
struct StreamConfig {
    int block_ms = 100;     // hop: new audio per inference
    int crossfade_ms = 40;  // SOLA crossfade (buffer capped at 40 ms)
    int extra_ms = 600;     // left context given to the neural models
    int lookahead_ms = 0;   // right context (future audio) beyond the decoded region

    // Throws std::invalid_argument with a precise message.
    void validate() const;
    std::string to_string() const;
    bool operator==(const StreamConfig&) const = default;
};

inline constexpr int kFrameMs = 10;
inline constexpr int kSolaSearchFrames = 1;
inline constexpr int kSolaBufferMaxFrames = 4;

// Frame geometry of one inference window (10 ms frames).
struct StreamGeometry {
    int block = 0;          // H
    int crossfade = 0;      // C
    int sola_buffer = 0;    // Cb = min(C, 4)
    int sola_search = kSolaSearchFrames;  // S
    int extra = 0;          // E
    int lookahead = 0;      // L
    int frames = 0;         // T = E + C + S + H + L (window)
    int skip_head = 0;      // E
    int return_length = 0;  // H + Cb + S

    // Delay from input to output caused by right context (C + S + L frames), in frames.
    int lookahead_frames() const { return crossfade + sola_search + lookahead; }
    std::string tag() const;  // "T75_s60_r15", matches the Python export file names
};

StreamGeometry make_geometry(const StreamConfig& cfg);

// Named presets shared with the Python importer.
StreamConfig preset(const std::string& name);  // low_latency | balanced | quality

}  // namespace xr::rvc
