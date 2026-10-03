#include "audio/simulated_stream.h"

#include <chrono>
#include <thread>

namespace xr::audio {

SimulatedDuplex::SimulatedDuplex(AudioCallbacks& callbacks, std::vector<float> input, SimulationConfig config)
    : cb_(callbacks), input_(std::move(input)), cfg_(config) {}

void SimulatedDuplex::run(double seconds) {
    const int in_period = cfg_.input_rate * cfg_.period_ms / 1000;
    const double out_period = cfg_.output_rate * cfg_.period_ms / 1000.0 * (1.0 + cfg_.output_clock_ppm * 1e-6);
    const int periods = static_cast<int>(seconds * 1000.0 / cfg_.period_ms);
    std::vector<float> in_buf(static_cast<size_t>(in_period));
    std::vector<float> out_buf(static_cast<size_t>(out_period) + 2);
    size_t pos = 0;
    double out_acc = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (int k = 0; k < periods; ++k) {
        if (cfg_.realtime) {
            std::this_thread::sleep_until(start + std::chrono::microseconds(static_cast<long long>(k) * cfg_.period_ms * 1000));
        }
        for (int i = 0; i < in_period; ++i) in_buf[static_cast<size_t>(i)] = pos < input_.size() ? input_[pos++] : (++pos, 0.0f);
        cb_.on_capture(in_buf.data(), static_cast<uint32_t>(in_period));
        out_acc += out_period;
        const auto n = static_cast<uint32_t>(out_acc);
        out_acc -= n;
        cb_.on_playback(out_buf.data(), n);
        recorded_.insert(recorded_.end(), out_buf.begin(), out_buf.begin() + n);
    }
}

}  // namespace xr::audio
