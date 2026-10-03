#include "rvc/stream_config.h"

#include <algorithm>
#include <stdexcept>

#include <spdlog/fmt/fmt.h>

namespace xr::rvc {

void StreamConfig::validate() const {
    auto check = [](const char* name, int v) {
        if (v < 0 || v % kFrameMs != 0) {
            throw std::invalid_argument(fmt::format("{}={} ms must be a non-negative multiple of {} ms", name, v,
                                                    kFrameMs));
        }
    };
    check("block_ms", block_ms);
    check("crossfade_ms", crossfade_ms);
    check("extra_ms", extra_ms);
    if (block_ms < 2 * kFrameMs) throw std::invalid_argument("block_ms must be at least 20 ms");
    if (crossfade_ms < kFrameMs) throw std::invalid_argument("crossfade_ms must be at least 10 ms");
}

std::string StreamConfig::to_string() const {
    return fmt::format("block {} ms, crossfade {} ms, extra {} ms", block_ms, crossfade_ms, extra_ms);
}

std::string StreamGeometry::tag() const { return fmt::format("T{}_s{}_r{}", frames, skip_head, return_length); }

StreamGeometry make_geometry(const StreamConfig& cfg) {
    cfg.validate();
    StreamGeometry g;
    g.block = cfg.block_ms / kFrameMs;
    g.crossfade = cfg.crossfade_ms / kFrameMs;
    g.sola_buffer = std::min(g.crossfade, kSolaBufferMaxFrames);
    g.extra = cfg.extra_ms / kFrameMs;
    g.frames = g.extra + g.crossfade + g.sola_search + g.block;
    g.skip_head = g.extra;
    g.return_length = g.block + g.sola_buffer + g.sola_search;
    return g;
}

StreamConfig preset(const std::string& name) {
    if (name == "low_latency") return {60, 30, 400};
    if (name == "balanced") return {100, 40, 600};
    if (name == "quality") return {200, 60, 1000};
    throw std::invalid_argument("unknown stream preset '" + name + "' (low_latency|balanced|quality)");
}

}  // namespace xr::rvc
