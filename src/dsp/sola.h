#pragma once

#include <span>
#include <vector>

namespace xr::dsp {

// Synchronized overlap-add with crossfade (SOLA), as used by RVC's realtime GUI
// (originally from DDSP-SVC). Each inference returns `block + buffer + search`
// samples. The offset in [0, search] that best correlates (normalised
// cross-correlation) with the tail kept from the previous block is chosen, the
// first `buffer` samples are crossfaded (sin^2 fade-in / cos^2 fade-out), `block`
// samples are emitted and the next `buffer` samples become the new tail.
// No allocation after construction.
class Sola {
public:
    Sola(int block, int buffer, int search);

    int block() const { return block_; }
    int buffer() const { return buffer_; }
    int search() const { return search_; }
    int input_size() const { return block_ + buffer_ + search_; }

    // `in` must hold input_size() samples, `out` block() samples. Returns the chosen offset.
    int process(std::span<const float> in, std::span<float> out);
    void reset();

private:
    int block_, buffer_, search_;
    std::vector<float> tail_;
    std::vector<float> fade_in_;
};

// Loudness envelope matching ("RMS mix rate"), as upstream realtime RVC:
//   rms frames: 40 ms window, 10 ms hop, centered with zero padding (librosa.feature.rms)
//   envelopes linearly interpolated to sample rate (align_corners)
//   out *= (rms_in / max(rms_out, 1e-3)) ^ (1 - rate)
// rate = 1 leaves the output untouched; rate = 0 follows the input loudness fully.
class RmsMixer {
public:
    // `frame` / `hop` in samples of the respective signal (40 ms / 10 ms).
    void apply(std::span<const float> input, int in_hop, std::span<float> output, int out_hop, float rate);

    // Exposed for tests: librosa-compatible RMS frames (center=True, constant padding).
    static void rms_frames(std::span<const float> x, int hop, std::vector<float>& out);

private:
    std::vector<float> rms_in_, rms_out_;
};

}  // namespace xr::dsp
