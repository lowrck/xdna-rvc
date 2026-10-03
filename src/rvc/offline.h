#pragma once

#include <functional>
#include <vector>

#include "rvc/stream_processor.h"
#include "util/latency_stats.h"

namespace xr::rvc {

struct OfflineResult {
    std::vector<float> audio;   // at processor output rate, aligned with the input
    int sample_rate = 0;
    int hops = 0;
    double wall_seconds = 0.0;  // processing time (excludes model loading)
    double audio_seconds = 0.0;
    // Per-stage timing statistics over all hops (ms).
    StatsSummary resample_in, content, pitch, index, generator, post, total;
    double realtime_factor() const { return audio_seconds > 0 ? wall_seconds / audio_seconds : 0.0; }
};

// Feeds `input` (mono, at processor input rate) through the streaming processor hop by
// hop — exactly the realtime code path — then removes the algorithmic delay so the
// output is time-aligned with the input. `progress` (optional) receives 0..1.
OfflineResult convert_offline(StreamProcessor& processor, const std::vector<float>& input,
                              const VoiceParamsSnapshot& params,
                              const std::function<void(double)>& progress = {});

}  // namespace xr::rvc
