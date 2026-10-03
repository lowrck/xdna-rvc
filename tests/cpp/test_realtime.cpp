#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "audio/realtime_engine.h"
#include "audio/simulated_stream.h"
#include "audio/spsc_ring.h"

using namespace xr;
using namespace xr::audio;

TEST_CASE("SPSC ring: wraparound, partial reads/writes, discard") {
    SpscRing r(10);  // rounds up to 16 (15 usable)
    CHECK(r.capacity() == 15);
    std::vector<float> a(12), b(12);
    for (int i = 0; i < 12; ++i) a[i] = float(i);
    CHECK(r.write(a) == 12);
    CHECK(r.write(a) == 3);  // full
    CHECK(r.read_available() == 15);
    CHECK(r.read(std::span<float>(b.data(), 10)) == 10);
    CHECK(b[9] == 9.0f);
    CHECK(r.write(a) == 10);  // wraps
    CHECK(r.discard(3) == 3);
    std::vector<float> c(20);
    const size_t n = r.read(c);
    CHECK(n == 12);
    CHECK(c[0] == 1.0f);   // a[1] (third of the 3 written in the second call... after discard)
    CHECK(c[2] == 0.0f);   // start of the wrapped write
    CHECK(c[11] == 9.0f);
    CHECK(r.read_available() == 0);
}

TEST_CASE("SPSC ring: concurrent producer/consumer preserves order") {
    SpscRing r(1024);
    constexpr int kTotal = 2'000'000;
    std::thread producer([&] {
        float buf[37];
        int next = 0;
        while (next < kTotal) {
            const int n = std::min(37, kTotal - next);
            for (int i = 0; i < n; ++i) buf[i] = float(next + i);
            const size_t w = r.write(std::span<const float>(buf, size_t(n)));
            next += int(w);
        }
    });
    int expected = 0;
    bool ok = true;
    float buf[53];
    while (expected < kTotal) {
        const size_t n = r.read(buf);
        for (size_t i = 0; i < n; ++i) ok = ok && (buf[i] == float(expected++));
    }
    producer.join();
    CHECK(ok);
}

namespace {

// Passthrough processor with a configurable per-hop cost, for scheduling tests.
class FakeProcessor final : public rvc::HopProcessor {
public:
    FakeProcessor(int in_rate, int out_rate, int hop_ms, double cost_ms)
        : cost_ms_(cost_ms), hop_in_(in_rate * hop_ms / 1000), hop_out_(out_rate * hop_ms / 1000), hop_ms_(hop_ms) {}
    int hop_input_samples() const override { return hop_in_; }
    int hop_output_samples() const override { return hop_out_; }
    double hop_seconds() const override { return hop_ms_ / 1000.0; }
    double algorithmic_latency_seconds() const override { return 0.0; }
    void reset() override {}
    void process(std::span<const float> in, std::span<float> out, const rvc::VoiceParamsSnapshot&,
                 rvc::HopTimings*) override {
        if (cost_ms_ > 0) std::this_thread::sleep_for(std::chrono::microseconds(long(cost_ms_ * 1000)));
        for (size_t i = 0; i < out.size(); ++i) out[i] = in[std::min(i, in.size() - 1)];
    }
    double cost_ms_;

private:
    int hop_in_, hop_out_, hop_ms_;
};

std::vector<float> ramp_signal(int rate, double seconds) {
    std::vector<float> x(size_t(rate * seconds));
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.3f * std::sin(2 * 3.14159265 * 220.0 * double(i) / rate);
    return x;
}

}  // namespace

TEST_CASE("engine: fast processing gives no underruns and a measured latency near prefill + hop") {
    EngineOptions o;
    o.safety_ms = 30;
    o.worker_priority = ThreadPriority::Normal;
    RealtimeEngine eng([](int i, int out) { return std::make_unique<FakeProcessor>(i, out, 40, 2.0); }, 48000, 48000,
                       10.0, 10.0, o);
    eng.start();
    SimulatedDuplex sim(eng, ramp_signal(48000, 3.0), {48000, 48000, 10, 0, true});
    sim.run(3.0);
    eng.stop();
    const auto s = eng.snapshot();
    CHECK(s.hops >= 70);
    // The first callback may race the worker start; afterwards there must be none.
    CHECK(s.underrun_events <= 1);
    CHECK(s.overrun_samples == 0);
    CHECK(s.worker_late_events == 0);
    // hop 40 + proc ~2 + ring ~(prefill - hop - proc) + devices 20 => ~ 40 + 30 + 20 = 90 ms
    CHECK(s.latency_ms.median > 60.0);
    CHECK(s.latency_ms.median < 120.0);
    // Output is the delayed input: not silent after the start-up.
    const auto& rec = sim.recorded();
    double e = 0;
    for (size_t i = rec.size() / 2; i < rec.size(); ++i) e += rec[i] * rec[i];
    CHECK(e > 100.0);
}

TEST_CASE("engine: processing slower than realtime is detected (underruns and late worker)") {
    EngineOptions o;
    o.safety_ms = 10;
    o.worker_priority = ThreadPriority::Normal;
    RealtimeEngine eng([](int i, int out) { return std::make_unique<FakeProcessor>(i, out, 20, 35.0); }, 48000, 48000,
                       10.0, 10.0, o);
    eng.start();
    SimulatedDuplex sim(eng, ramp_signal(48000, 2.0), {48000, 48000, 10, 0, true});
    sim.run(2.0);
    eng.stop();
    const auto s = eng.snapshot();
    CHECK(s.underrun_events > 10);
    CHECK(s.worker_late_events > 0);
    CHECK(s.total.median > 30.0);
}

TEST_CASE("engine: separate device clocks are followed by drift compensation") {
    EngineOptions o;
    o.safety_ms = 20;
    o.worker_priority = ThreadPriority::Normal;
    RealtimeEngine eng([](int i, int out) { return std::make_unique<FakeProcessor>(i, out, 20, 1.0); }, 48000, 48000,
                       5.0, 5.0, o);
    eng.start();
    // Output clock 600 ppm fast (much worse than typical hardware): ~29 extra samples/s
    // are consumed. Without compensation the ring would drain by ~290 samples in 10 s.
    SimulatedDuplex sim(eng, ramp_signal(48000, 10.0), {48000, 48000, 10, 600.0, true});
    sim.run(10.0);
    eng.stop();
    const auto s = eng.snapshot();
    CHECK(s.drift_corrections > 40);
    CHECK(s.underrun_events <= 1);
}

TEST_CASE("engine: inference errors produce silence and are counted, not crashes") {
    class Throwing final : public rvc::HopProcessor {
    public:
        int hop_input_samples() const override { return 480; }
        int hop_output_samples() const override { return 480; }
        double hop_seconds() const override { return 0.01; }
        double algorithmic_latency_seconds() const override { return 0; }
        void reset() override {}
        void process(std::span<const float>, std::span<float>, const rvc::VoiceParamsSnapshot&, rvc::HopTimings*) override {
            throw std::runtime_error("simulated NPU timeout");
        }
    };
    EngineOptions o;
    o.worker_priority = ThreadPriority::Normal;
    RealtimeEngine eng([](int, int) { return std::make_unique<Throwing>(); }, 48000, 48000, 0, 0, o);
    eng.start();
    SimulatedDuplex sim(eng, ramp_signal(48000, 0.5), {48000, 48000, 10, 0, true});
    sim.run(0.5);
    eng.stop();
    const auto s = eng.snapshot();
    CHECK(s.inference_errors > 10);
    CHECK(s.last_error == "simulated NPU timeout");
    for (float v : sim.recorded()) REQUIRE(v == 0.0f);
}

TEST_CASE("engine: latency added by an underrun is trimmed back during silence") {
    // Speech for 2 s, then silence. One 150 ms stall during speech causes an underrun,
    // which permanently queues ~150 ms of extra audio unless the engine trims it.
    class Stalling final : public rvc::HopProcessor {
    public:
        int hop_input_samples() const override { return 960; }
        int hop_output_samples() const override { return 960; }
        double hop_seconds() const override { return 0.02; }
        double algorithmic_latency_seconds() const override { return 0; }
        void reset() override {}
        void process(std::span<const float> in, std::span<float> out, const rvc::VoiceParamsSnapshot&,
                     rvc::HopTimings*) override {
            if (++n_ == 80) std::this_thread::sleep_for(std::chrono::milliseconds(150));
            std::copy(in.begin(), in.end(), out.begin());
        }
        int n_ = 0;
    };
    EngineOptions o;
    o.safety_ms = 20;
    o.worker_priority = ThreadPriority::Normal;
    RealtimeEngine eng([](int, int) { return std::make_unique<Stalling>(); }, 48000, 48000, 0, 0, o);
    eng.start();
    auto speech = ramp_signal(48000, 2.0);
    speech.resize(48000 * 6, 0.0f);  // then 4 s of silence
    SimulatedDuplex sim(eng, speech, {48000, 48000, 10, 0, true});
    sim.run(6.0);
    eng.stop();
    const auto s = eng.snapshot();
    CHECK(s.underrun_events >= 1);
    CHECK(s.latency_trims + s.worker_late_events >= 1);  // recovered by trimming and/or backlog drop
    // Latency at the end is back near the pre-stall value (hop 20 + safety 20 + ring).
    CHECK(s.latency_history.back() < s.latency_ms.max - 100.0);
}
