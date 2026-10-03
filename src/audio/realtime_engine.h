#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "audio/audio_backend.h"
#include "audio/spsc_ring.h"
#include "rvc/hop_processor.h"
#include "rvc/stream_processor.h"
#include "rvc/voice_params.h"
#include "util/latency_stats.h"
#include "util/thread_priority.h"

namespace xr::audio {

// What the listener hears when converted audio is not ready in time.
enum class UnderrunPolicy {
    Silence,  // short fade to silence (default)
    Bypass,   // the dry input (only when input and output rates are equal)
};

// Builds the hop processor once the device sample rates are known.
using ProcessorFactory = std::function<std::unique_ptr<rvc::HopProcessor>(int input_rate, int output_rate)>;

struct EngineOptions {
    UnderrunPolicy underrun_policy = UnderrunPolicy::Silence;
    // Output pre-fill beyond one hop, to absorb processing time and scheduling jitter.
    // Larger = fewer underruns, more latency. -1 = automatic: 1.25 x the slowest warm-up
    // hop + 10 ms. See docs/architecture.md "Buffering".
    int safety_ms = -1;
    int max_input_backlog_hops = 3;    // beyond this the worker drops old input to catch up
    bool drift_compensation = true;    // +-1 sample per hop to follow separate device clocks
    ThreadPriority worker_priority = ThreadPriority::Realtime;
    int stats_window = 500;            // hops in rolling statistics
};

// Counters written by audio threads / worker, read by anyone (relaxed atomics).
struct EngineCounters {
    std::atomic<uint64_t> hops{0};
    std::atomic<uint64_t> underrun_events{0};    // playback found fewer samples than requested
    std::atomic<uint64_t> underrun_samples{0};
    std::atomic<uint64_t> overrun_samples{0};    // capture ring full: input dropped
    std::atomic<uint64_t> worker_late_events{0}; // worker fell behind, old input dropped
    std::atomic<uint64_t> inference_errors{0};
    std::atomic<uint64_t> drift_corrections{0};
};

// Snapshot for UIs, published by the worker.
struct EngineSnapshot {
    bool running = false;
    std::string device_description;
    std::string worker_priority;
    int hop_ms = 0;
    int safety_ms = 0;  // effective output safety buffer
    double hop_budget_ms = 0;
    StatsSummary resample_in, content, pitch, index, generator, post, total;
    StatsSummary latency_ms;          // measured per hop (see RealtimeEngine docs)
    double input_ring_ms = 0, output_ring_ms = 0;
    double input_device_ms = 0, output_device_ms = 0, algorithmic_ms = 0;
    uint64_t hops = 0, underrun_events = 0, underrun_samples = 0, overrun_samples = 0, worker_late_events = 0,
             inference_errors = 0, drift_corrections = 0;
    std::string last_error;
    std::vector<float> total_history;    // per-hop processing time, ms
    std::vector<float> latency_history;  // per-hop measured latency, ms
};

// Realtime voice conversion:
//
//   capture callback -> input ring -> worker: StreamProcessor::process -> output ring -> playback callback
//
// Audio callbacks only copy to/from lock-free rings, update atomics and wake the worker
// (std::atomic::notify_one); no allocation, locks, I/O, logging or inference.
//
// Latency of every hop k is measured from observed quantities:
//   hop + processing_time_k + output_ring_level_before_push_k + algorithmic delay
//   + input device buffer + output device buffer
// i.e. how long the first sample of the hop waits from the microphone to the speaker.
class RealtimeEngine final : public AudioCallbacks {
public:
    RealtimeEngine(ProcessorFactory factory, AudioBackend& backend, const StreamParams& stream, EngineOptions options);
    // Externally driven callbacks (simulated audio for tests and offline realtime simulation).
    RealtimeEngine(ProcessorFactory factory, int input_rate, int output_rate, double input_device_ms,
                   double output_device_ms, EngineOptions options);
    ~RealtimeEngine() override;

    void start();
    void stop();
    bool running() const { return running_.load(); }

    rvc::VoiceParams& params() { return params_; }
    const EngineCounters& counters() const { return counters_; }
    EngineSnapshot snapshot() const;
    rvc::HopProcessor& processor() { return *processor_; }

    // AudioCallbacks (audio threads)
    void on_capture(const float* mono, uint32_t frames) override;
    void on_playback(float* mono, uint32_t frames) override;

private:
    void init(const ProcessorFactory& factory, int in_rate, int out_rate);
    void worker_main();
    void publish(bool force);

    EngineOptions opts_;
    std::unique_ptr<AudioStream> stream_;
    std::unique_ptr<rvc::HopProcessor> processor_;
    rvc::VoiceParams params_;
    EngineCounters counters_;
    int in_rate_ = 0, out_rate_ = 0;
    double in_dev_ms_ = 0, out_dev_ms_ = 0;
    size_t prefill_ = 0;
    int effective_safety_ms_ = 0;

    std::unique_ptr<SpscRing> in_ring_;
    std::unique_ptr<SpscRing> out_ring_;
    std::unique_ptr<SpscRing> dry_ring_;  // bypass source (same-rate devices only)
    std::atomic<uint32_t> wake_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};
    std::thread worker_;
    float last_out_ = 0.0f;               // playback thread only

    // worker-owned
    std::vector<float> hop_in_, hop_out_, stretched_;
    RollingStats s_rin_, s_c_, s_p_, s_i_, s_g_, s_post_, s_t_, s_lat_;
    double level_avg_ = -1.0;
    double drift_target_ = 0.0;
    uint64_t hops_seen_ = 0;
    static constexpr uint64_t kDriftLearnHops = 50;
    std::string priority_desc_;

    mutable std::mutex snap_mu_;
    EngineSnapshot snap_;
};

const char* to_string(UnderrunPolicy p);

}  // namespace xr::audio
