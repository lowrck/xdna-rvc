#include "audio/realtime_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <spdlog/fmt/fmt.h>

#include "util/error.h"
#include "util/log.h"

namespace xr::audio {

const char* to_string(UnderrunPolicy p) { return p == UnderrunPolicy::Bypass ? "bypass" : "silence"; }

RealtimeEngine::RealtimeEngine(ProcessorFactory factory, AudioBackend& backend, const StreamParams& stream,
                               EngineOptions options)
    : opts_(std::move(options)),
      s_rin_(opts_.stats_window), s_c_(opts_.stats_window), s_p_(opts_.stats_window), s_i_(opts_.stats_window),
      s_g_(opts_.stats_window), s_post_(opts_.stats_window), s_t_(opts_.stats_window), s_lat_(opts_.stats_window) {
    stream_ = backend.open(stream, *this);
    in_dev_ms_ = stream_->input_buffer_ms();
    out_dev_ms_ = stream_->output_buffer_ms();
    XR_LOG_INFO("audio: {}", stream_->description());
    init(factory, stream_->input_rate(), stream_->output_rate());
}

RealtimeEngine::RealtimeEngine(ProcessorFactory factory, int input_rate, int output_rate, double input_device_ms,
                               double output_device_ms, EngineOptions options)
    : opts_(std::move(options)),
      in_dev_ms_(input_device_ms),
      out_dev_ms_(output_device_ms),
      s_rin_(opts_.stats_window), s_c_(opts_.stats_window), s_p_(opts_.stats_window), s_i_(opts_.stats_window),
      s_g_(opts_.stats_window), s_post_(opts_.stats_window), s_t_(opts_.stats_window), s_lat_(opts_.stats_window) {
    init(factory, input_rate, output_rate);
}

void RealtimeEngine::init(const ProcessorFactory& factory, int in_rate, int out_rate) {
    in_rate_ = in_rate;
    out_rate_ = out_rate;
    processor_ = factory(in_rate, out_rate);
    const size_t hop_in = static_cast<size_t>(processor_->hop_input_samples());
    const size_t hop_out = static_cast<size_t>(processor_->hop_output_samples());
    // Sized for the largest pre-fill we allow (one hop + up to three hops of safety).
    const size_t max_prefill = hop_out * 4 + static_cast<size_t>(std::max(0, opts_.safety_ms)) * out_rate / 1000;
    in_ring_ = std::make_unique<SpscRing>(hop_in * static_cast<size_t>(opts_.max_input_backlog_hops + 4));
    out_ring_ = std::make_unique<SpscRing>(max_prefill + hop_out * 6);
    prefill_ = hop_out;
    if (opts_.underrun_policy == UnderrunPolicy::Bypass) {
        if (in_rate != out_rate) {
            XR_LOG_WARN("bypass-on-underrun needs equal input/output rates ({} vs {}); using silence", in_rate, out_rate);
            opts_.underrun_policy = UnderrunPolicy::Silence;
        } else {
            dry_ring_ = std::make_unique<SpscRing>(max_prefill + hop_out * 6);
        }
    }
    hop_in_.assign(hop_in, 0.0f);
    hop_out_.assign(hop_out, 0.0f);
    stretched_.assign(hop_out + 2, 0.0f);
    XR_LOG_INFO("realtime engine: hop {} ms, underrun policy {}", processor_->hop_seconds() * 1000,
                to_string(opts_.underrun_policy));
}

RealtimeEngine::~RealtimeEngine() { stop(); }

void RealtimeEngine::start() {
    if (running_) return;
    // Warm-up: the first inference of each session is slower (lazy allocations, kernel
    // selection). Run it now, before audio starts, so it cannot cause an underrun.
    double slowest_ms = 0.0;
    {
        rvc::VoiceParamsSnapshot p = params_.snapshot();
        for (int i = 0; i < 3; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            try {
                processor_->process(hop_in_, hop_out_, p, nullptr);
            } catch (const std::exception& e) {
                XR_LOG_WARN("warm-up inference failed: {}", e.what());
            }
            // The first call includes one-time initialisation; size the buffer from the rest.
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (i > 0) slowest_ms = std::max(slowest_ms, ms);
        }
    }
    const size_t hop_out = hop_out_.size();
    effective_safety_ms_ = opts_.safety_ms >= 0 ? opts_.safety_ms
                                                : static_cast<int>(std::ceil(1.25 * slowest_ms)) + 10;
    const int max_safety_ms = static_cast<int>(processor_->hop_seconds() * 3000.0) + std::max(0, opts_.safety_ms);
    effective_safety_ms_ = std::min(effective_safety_ms_, max_safety_ms);
    prefill_ = hop_out + static_cast<size_t>(effective_safety_ms_) * static_cast<size_t>(out_rate_) / 1000;
    XR_LOG_INFO("warm-up: slowest hop {:.1f} ms; output pre-fill {:.1f} ms (hop + {} ms safety{})", slowest_ms,
                1000.0 * prefill_ / out_rate_, effective_safety_ms_, opts_.safety_ms < 0 ? ", automatic" : "");
    processor_->reset();
    // Pre-fill the output with silence: playback needs data while the first hop is captured
    // and processed.
    std::vector<float> zeros(prefill_, 0.0f);
    out_ring_->discard(out_ring_->read_available());
    out_ring_->write(zeros);
    in_ring_->discard(in_ring_->read_available());
    level_avg_ = -1.0;
    hops_seen_ = 0;
    stop_ = false;
    running_ = true;
    worker_ = std::thread([this] { worker_main(); });
    if (stream_) {
        try {
            stream_->start();
        } catch (...) {
            stop();
            throw;
        }
    }
    publish(true);
}

void RealtimeEngine::stop() {
    if (!running_ && !worker_.joinable()) return;
    if (stream_) stream_->stop();
    stop_ = true;
    wake_.fetch_add(1);
    wake_.notify_one();
    if (worker_.joinable()) worker_.join();
    running_ = false;
    publish(true);
}

void RealtimeEngine::on_capture(const float* mono, uint32_t frames) {
    if (!running_.load(std::memory_order_relaxed)) return;
    const size_t n = in_ring_->write(std::span<const float>(mono, frames));
    if (n < frames) counters_.overrun_samples.fetch_add(frames - n, std::memory_order_relaxed);
    if (dry_ring_) dry_ring_->write(std::span<const float>(mono, frames));
    wake_.fetch_add(1, std::memory_order_release);
    wake_.notify_one();
}

void RealtimeEngine::on_playback(float* out, uint32_t frames) {
    if (!running_.load(std::memory_order_relaxed)) {
        std::fill(out, out + frames, 0.0f);
        return;
    }
    const size_t n = out_ring_->read(std::span<float>(out, frames));
    if (dry_ring_) {
        // Keep the dry signal roughly aligned with what is being played.
        const size_t backlog = dry_ring_->read_available();
        if (backlog > prefill_ + frames) dry_ring_->discard(backlog - prefill_);
    }
    if (n < frames) {
        counters_.underrun_events.fetch_add(1, std::memory_order_relaxed);
        counters_.underrun_samples.fetch_add(frames - n, std::memory_order_relaxed);
        size_t filled = n;
        if (dry_ring_) filled += dry_ring_->read(std::span<float>(out + n, frames - n));
        // Fade from the last played value to zero over 64 samples instead of a hard step.
        float v = filled > 0 ? out[filled - 1] : last_out_;
        for (size_t i = filled; i < frames; ++i) {
            v *= 0.92f;
            out[i] = v;
        }
    } else if (dry_ring_) {
        dry_ring_->discard(frames);
    }
    for (uint32_t i = 0; i < frames; ++i) out[i] = std::clamp(out[i], -1.0f, 1.0f);
    if (frames) last_out_ = out[frames - 1];
}

void RealtimeEngine::worker_main() {
    priority_desc_ = set_current_thread_priority(opts_.worker_priority);
    XR_LOG_INFO("inference worker priority: {}", priority_desc_);
    const size_t hop_in = hop_in_.size();
    const size_t hop_out = hop_out_.size();
    const double algorithmic_ms = processor_->algorithmic_latency_seconds() * 1000.0;
    const double hop_ms = processor_->hop_seconds() * 1000.0;
    int consecutive_errors = 0;
    using Clock = std::chrono::steady_clock;
    auto last_publish = Clock::now();

    while (!stop_.load()) {
        const uint32_t seen = wake_.load(std::memory_order_acquire);
        size_t avail = in_ring_->read_available();
        if (avail < hop_in) {
            wake_.wait(seen, std::memory_order_acquire);
            continue;
        }
        if (avail > hop_in * static_cast<size_t>(opts_.max_input_backlog_hops)) {
            // Fell behind (e.g. a slow inference or a stall): drop the oldest input so latency
            // stays bounded instead of growing forever.
            const size_t drop = (avail / hop_in - 1) * hop_in;
            in_ring_->discard(drop);
            counters_.worker_late_events.fetch_add(1, std::memory_order_relaxed);
        }
        in_ring_->read(hop_in_);

        const auto params = params_.snapshot();
        rvc::HopTimings t;
        const auto t0 = Clock::now();
        try {
            if (params.bypass && in_rate_ == out_rate_) {
                std::copy(hop_in_.begin(), hop_in_.end(), hop_out_.begin());
            } else {
                processor_->process(hop_in_, hop_out_, params, &t);
            }
            consecutive_errors = 0;
        } catch (const std::exception& e) {
            std::fill(hop_out_.begin(), hop_out_.end(), 0.0f);
            counters_.inference_errors.fetch_add(1, std::memory_order_relaxed);
            if (++consecutive_errors <= 3) XR_LOG_ERROR("inference failed: {}", e.what());
            std::lock_guard lock(snap_mu_);
            snap_.last_error = e.what();
        }
        const double proc_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // Clock drift between separate input/output devices: keep the output level near its
        // target by stretching/shrinking this hop by one sample (linear interpolation).
        const size_t level = out_ring_->read_available();
        std::span<const float> to_push(hop_out_);
        if (opts_.drift_compensation) {
            // The steady-state level depends on the (unknown) mean processing time, so the
            // target is learned over the first hops, then held; afterwards only clock drift
            // moves the average.
            level_avg_ = level_avg_ < 0 ? static_cast<double>(level) : 0.98 * level_avg_ + 0.02 * level;
            ++hops_seen_;
            if (hops_seen_ == kDriftLearnHops) drift_target_ = level_avg_;
            const double band = static_cast<double>(hop_out) / 8.0;
            int delta = 0;
            if (hops_seen_ > kDriftLearnHops) {
                if (level_avg_ > drift_target_ + band) delta = -1;
                else if (level_avg_ < drift_target_ - band) delta = +1;
            }
            // Latency creep after underruns: the missing samples stay queued. Recover quickly by
            // dropping part of a *silent* block (inaudible) rather than waiting for the
            // one-sample drift corrections.
            const double excess = level_avg_ - (drift_target_ + band);
            if (hops_seen_ > kDriftLearnHops && excess > static_cast<double>(hop_out) / 4.0) {
                double energy = 0.0;
                for (float v : hop_out_) energy += static_cast<double>(v) * v;
                const double rms_db = 10.0 * std::log10(energy / static_cast<double>(hop_out) + 1e-20);
                if (rms_db < -55.0) {
                    const size_t drop = std::min(static_cast<size_t>(excess), hop_out / 2);
                    to_push = std::span<const float>(hop_out_.data(), hop_out - drop);
                    level_avg_ -= static_cast<double>(drop);
                    counters_.latency_trims.fetch_add(1, std::memory_order_relaxed);
                    delta = 0;
                }
            }
            if (delta != 0) {
                const size_t n = hop_out + static_cast<size_t>(static_cast<long>(delta));
                const double step = static_cast<double>(hop_out - 1) / static_cast<double>(n - 1);
                for (size_t i = 0; i < n; ++i) {
                    const double pos = static_cast<double>(i) * step;
                    const size_t a = static_cast<size_t>(pos);
                    const size_t b = std::min(a + 1, hop_out - 1);
                    const float f = static_cast<float>(pos - static_cast<double>(a));
                    stretched_[i] = hop_out_[a] * (1.0f - f) + hop_out_[b] * f;
                }
                to_push = std::span<const float>(stretched_.data(), n);
                counters_.drift_corrections.fetch_add(1, std::memory_order_relaxed);
                level_avg_ -= delta;  // count the correction once
            }
        }
        out_ring_->write(to_push);
        counters_.hops.fetch_add(1, std::memory_order_relaxed);

        const double latency = hop_ms + proc_ms + 1000.0 * static_cast<double>(level) / out_rate_ + algorithmic_ms +
                               in_dev_ms_ + out_dev_ms_;
        s_rin_.push(t.resample_in);
        s_c_.push(t.content);
        s_p_.push(t.pitch);
        s_i_.push(t.index);
        s_g_.push(t.generator);
        s_post_.push(t.post);
        s_t_.push(proc_ms);
        s_lat_.push(latency);
        if (Clock::now() - last_publish > std::chrono::milliseconds(100)) {
            publish(false);
            last_publish = Clock::now();
        }
    }
}

void RealtimeEngine::publish(bool force) {
    std::unique_lock lock(snap_mu_, std::defer_lock);
    if (force) {
        lock.lock();
    } else if (!lock.try_lock()) {
        return;  // a reader holds it; publish next time rather than block the worker
    }
    snap_.running = running_.load();
    snap_.device_description = stream_ ? stream_->description() : "simulated audio";
    snap_.worker_priority = priority_desc_;
    snap_.hop_ms = processor_ ? static_cast<int>(processor_->hop_seconds() * 1000) : 0;
    snap_.hop_budget_ms = snap_.hop_ms;
    snap_.safety_ms = effective_safety_ms_;
    snap_.resample_in = s_rin_.summary();
    snap_.content = s_c_.summary();
    snap_.pitch = s_p_.summary();
    snap_.index = s_i_.summary();
    snap_.generator = s_g_.summary();
    snap_.post = s_post_.summary();
    snap_.total = s_t_.summary();
    snap_.latency_ms = s_lat_.summary();
    s_t_.history(snap_.total_history);
    s_lat_.history(snap_.latency_history);
    snap_.input_ring_ms = in_ring_ ? 1000.0 * in_ring_->read_available() / in_rate_ : 0;
    snap_.output_ring_ms = out_ring_ ? 1000.0 * out_ring_->read_available() / out_rate_ : 0;
    snap_.input_device_ms = in_dev_ms_;
    snap_.output_device_ms = out_dev_ms_;
    snap_.algorithmic_ms = processor_ ? processor_->algorithmic_latency_seconds() * 1000.0 : 0;
    snap_.hops = counters_.hops.load();
    snap_.underrun_events = counters_.underrun_events.load();
    snap_.underrun_samples = counters_.underrun_samples.load();
    snap_.overrun_samples = counters_.overrun_samples.load();
    snap_.worker_late_events = counters_.worker_late_events.load();
    snap_.inference_errors = counters_.inference_errors.load();
    snap_.drift_corrections = counters_.drift_corrections.load();
    snap_.latency_trims = counters_.latency_trims.load();
}

EngineSnapshot RealtimeEngine::snapshot() const {
    std::lock_guard lock(snap_mu_);
    return snap_;
}

}  // namespace xr::audio
