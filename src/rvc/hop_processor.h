#pragma once

#include <span>

#include "rvc/voice_params.h"

namespace xr::rvc {

struct HopTimings;

// One hop in, one hop out. Implemented by StreamProcessor (the RVC pipeline) and by
// test doubles; the realtime engine only depends on this interface.
class HopProcessor {
public:
    virtual ~HopProcessor() = default;
    virtual int hop_input_samples() const = 0;
    virtual int hop_output_samples() const = 0;
    virtual double hop_seconds() const = 0;
    virtual double algorithmic_latency_seconds() const = 0;
    virtual void reset() = 0;
    virtual void process(std::span<const float> in, std::span<float> out, const VoiceParamsSnapshot& params,
                         HopTimings* timings) = 0;
};

}  // namespace xr::rvc
