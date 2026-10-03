#pragma once

#include <vector>

#include "audio/audio_backend.h"

namespace xr::audio {

struct SimulationConfig {
    int input_rate = 48000;
    int output_rate = 48000;
    int period_ms = 10;            // device callback period
    double output_clock_ppm = 0;   // output device clock error relative to input (drift)
    bool realtime = true;          // pace callbacks with the wall clock (false: as fast as possible)
};

// Drives AudioCallbacks like a duplex device: every period, one capture callback with the
// next chunk of `input` (silence after the end) and one playback callback whose output is
// recorded. With realtime=true the timing matches real devices, so worker scheduling,
// underruns and latency behave as they would with hardware.
class SimulatedDuplex {
public:
    SimulatedDuplex(AudioCallbacks& callbacks, std::vector<float> input, SimulationConfig config);
    // Runs for `seconds` of simulated device time (blocking).
    void run(double seconds);
    const std::vector<float>& recorded() const { return recorded_; }

private:
    AudioCallbacks& cb_;
    std::vector<float> input_;
    SimulationConfig cfg_;
    std::vector<float> recorded_;
};

}  // namespace xr::audio
