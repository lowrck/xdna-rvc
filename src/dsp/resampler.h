#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace xr::dsp {

enum class ResamplerQuality {
    Fast,      // 8 zero crossings: lowest latency, mild aliasing near Nyquist
    Balanced,  // 16 zero crossings: default for realtime
    High,      // 32 zero crossings: offline conversion
};

// Streaming rational-ratio polyphase resampler (Kaiser-windowed sinc).
//
// Realtime-safe: after construction, process() performs no allocation, no locking
// and no I/O, provided `out` has room for max_output(in.size()) samples.
//
// Timing: output sample k corresponds to input time k * in_rate / out_rate. Because
// the filter is centered, producing it requires `lookahead_input()` future input
// samples, which is the resampler's streaming latency.
class Resampler {
public:
    Resampler(int in_rate, int out_rate, ResamplerQuality quality = ResamplerQuality::Balanced);

    int in_rate() const { return in_rate_; }
    int out_rate() const { return out_rate_; }
    bool is_passthrough() const { return up_ == down_; }

    // Upper bound of output samples produced by process() for `in_frames` input.
    size_t max_output(size_t in_frames) const;
    // Consumes all of `in`, writes produced samples to `out`; returns count written.
    size_t process(std::span<const float> in, std::span<float> out);
    void reset();

    // Input samples of lookahead (latency) introduced by the filter.
    int lookahead_input() const { return taps_ / 2; }
    double latency_seconds() const { return static_cast<double>(lookahead_input()) / in_rate_; }
    int taps_per_phase() const { return taps_; }

    // Whole-buffer conversion, latency-compensated: output length is
    // round(in.size() * out_rate / in_rate) and sample k aligns with input time k/out_rate.
    static std::vector<float> resample(std::span<const float> in, int in_rate, int out_rate,
                                       ResamplerQuality quality = ResamplerQuality::High);

private:
    static constexpr size_t kChunk = 4096;

    int in_rate_;
    int out_rate_;
    int up_;    // L
    int down_;  // M
    int taps_;  // per phase, even
    std::vector<float> coeffs_;  // up_ * taps_, phase-major
    std::vector<float> work_;    // history + current chunk
    size_t valid_ = 0;           // valid samples in work_
    int64_t pos_ = 0;            // index in work_ of floor(t) for the next output
    int phase_ = 0;              // (k * down_) mod up_
};

}  // namespace xr::dsp
