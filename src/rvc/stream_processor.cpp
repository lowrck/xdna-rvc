#include "rvc/stream_processor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>

#include <spdlog/fmt/fmt.h>

#include "inference/ort_runtime.h"
#include "util/error.h"
#include "util/log.h"

namespace xr::rvc {

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

float db_to_gain(float db) { return std::pow(10.0f, db / 20.0f); }

// RVC coarse pitch bin (upstream get_f0_post); np.rint/torch.round round half to even,
// as does std::nearbyint in the default rounding mode.
int64_t coarse_pitch(float f0) {
    static const double mel_min = 1127.0 * std::log(1.0 + 50.0 / 700.0);
    static const double mel_max = 1127.0 * std::log(1.0 + 1100.0 / 700.0);
    double mel = 1127.0 * std::log(1.0 + static_cast<double>(f0) / 700.0);
    if (mel > 0) mel = (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0;
    mel = std::clamp(mel, 1.0, 255.0);
    return static_cast<int64_t>(std::nearbyint(mel));
}

std::string cache_key_for(const std::filesystem::path& model, const std::string& suffix) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(model, ec);
    std::string stem = model.stem().string();
    std::replace_if(stem.begin(), stem.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
    return fmt::format("{}_{}_{:08x}", stem, suffix,
                       static_cast<uint32_t>(std::hash<std::string>{}(model.string() + std::to_string(size))));
}

}  // namespace

StreamProcessor::StreamProcessor(OrtRuntime& runtime, const ModelInfo& model, const ProcessorOptions& options)
    : runtime_(runtime),
      model_(model),
      opts_(options),
      geom_(make_geometry(options.stream)),
      in_resampler_(options.input_rate, ContentEncoder::kSampleRate, options.resampler_quality),
      out_resampler_(model.sample_rate, options.output_rate, options.resampler_quality),
      sola_(geom_.block * model.upsample_factor, geom_.sola_buffer * model.upsample_factor,
            geom_.sola_search * model.upsample_factor),
      rng_(options.seed ? static_cast<std::mt19937::result_type>(options.seed) : std::random_device{}()) {
    if (opts_.input_rate % 100 != 0 || opts_.output_rate % 100 != 0) {
        throw UserError(fmt::format("sample rates must be multiples of 100 Hz (got input {}, output {})",
                                    opts_.input_rate, opts_.output_rate),
                        "Use 44100 or 48000 Hz devices.");
    }
    const GeneratorVariant* variant = model_.find_generator(geom_);
    if (!variant) {
        std::string have;
        for (const auto& g : model_.generators) {
            have += fmt::format("\n  - {} ms block / {} ms crossfade / {} ms extra ({})", g.stream.block_ms,
                                g.stream.crossfade_ms, g.stream.extra_ms, g.path.filename().string());
        }
        throw UserError(fmt::format("voice '{}' has no generator exported for {} ({}). Available:{}", model_.name,
                                    opts_.stream.to_string(), geom_.tag(), have),
                        fmt::format("Export it with: python tools/convert_rvc.py <voice.pth> --name {} --stream {},{},{}",
                                    model_.name, opts_.stream.block_ms, opts_.stream.crossfade_ms,
                                    opts_.stream.extra_ms));
    }
    if (!variant->validated) {
        XR_LOG_WARN("generator {} did not pass all conversion checks; output may be wrong",
                    variant->path.filename().string());
    }
    hop_in_ = geom_.block * opts_.input_rate / 100;
    hop_out_ = geom_.block * opts_.output_rate / 100;
    zc_model_ = model_.upsample_factor;
    window16k_ = geom_.frames * 160;

    const auto& b = opts_.backends;
    auto base_request = [&](const std::string& stage, const std::filesystem::path& path) {
        SessionRequest r;
        r.stage = stage;
        r.model_path = path;
        r.intra_op_threads = b.intra_op_threads;
        r.collect_evidence = b.collect_evidence;
        r.keep_profiling = b.keep_profiling;
        r.ai_analyzer = b.ai_analyzer;
        r.diagnostics_dir = b.diagnostics_dir;
        r.cache_dir = b.cache_dir / "vaip";
        r.static_model_dir = b.cache_dir / "static_models";
        return r;
    };
    // Attach the precompiled XDNA 2 model for this stage/shape, if compile_xdna.py made one.
    auto attach_xdna = [&](SessionRequest& r, const std::filesystem::path& source) {
        const XdnaEntry* e = model_.find_xdna(r.stage, source, r.free_dims);
        if (!e) return;
        r.xdna_model_path = e->static_model;
        r.cache_dir = e->cache_dir;
        r.cache_key = e->cache_key;
        r.xdna_config_file = e->config_file;
        if (!e->compiled) {
            XR_LOG_WARN("[{}] XDNA model {} was prepared but not compiled; run tools/compile_xdna.py on the NPU "
                        "machine (BF16 cannot be compiled by the deployment runtime)", r.stage,
                        e->static_model.filename().string());
        }
        if (e->accuracy_passed && !*e->accuracy_passed) {
            r.xdna_auto_veto = fmt::format("its BF16 output differs from FP32 (rel. RMS {:.3f}) beyond tolerance",
                                           e->max_rel_rms);
        }
        if (!e->onnxruntime_version.empty() && e->onnxruntime_version != runtime_.version()) {
            XR_LOG_WARN("[{}] XDNA cache was compiled with ONNX Runtime {} but {} is loaded; AMD advises against "
                        "reusing caches across EP versions - recompile with tools/compile_xdna.py",
                        r.stage, e->onnxruntime_version, runtime_.version());
        }
    };
    auto open_opts = [&](BackendKind k) {
        OpenOptions o;
        o.backend = k;
        o.allow_cpu_fallback = b.allow_cpu_fallback;
        return o;
    };

    {
        auto r = base_request("content_encoder", model_.content_encoder);
        r.free_dims["samples"] = window16k_;
        r.cache_key = cache_key_for(model_.content_encoder, std::to_string(window16k_));
        attach_xdna(r, model_.content_encoder);
        content_ = std::make_unique<ContentEncoder>(runtime_, r, open_opts(b.content));
        if (content_->feature_dim() != model_.feature_dim) {
            throw UserError(fmt::format("content encoder {} produces {}-dim features but voice '{}' (RVC {}) needs {}",
                                        model_.content_encoder.string(), content_->feature_dim(), model_.name,
                                        model_.rvc_version, model_.feature_dim));
        }
        content_->prepare_fixed(window16k_);
    }
    if (model_.uses_f0) {
        pitch_segment_ = RmvpePitch::realtime_segment_samples(geom_.block * 160);
        if (pitch_segment_ > window16k_) {
            throw UserError(fmt::format("stream window ({} ms) is shorter than the RMVPE analysis segment ({} ms)",
                                        geom_.frames * 10, pitch_segment_ / 16),
                            "Increase extra_ms.");
        }
        auto r = base_request("rmvpe", model_.rmvpe);
        const int frames = RmvpePitch::frames_for_samples(pitch_segment_);
        r.free_dims["frames"] = frames;
        r.cache_key = cache_key_for(model_.rmvpe, std::to_string(frames));
        attach_xdna(r, model_.rmvpe);
        rmvpe_ = std::make_unique<RmvpePitch>(runtime_, r, open_opts(b.pitch), model_.rmvpe_mel_basis);
        rmvpe_->prepare_fixed(pitch_segment_);
        if (rmvpe_->fixed_frames() - 4 < geom_.block) {
            throw UserError("RMVPE segment too short for the block size");
        }
    }
    {
        auto r = base_request("generator", variant->path);
        r.cache_key = cache_key_for(variant->path, geom_.tag());
        attach_xdna(r, variant->path);
        generator_ = std::make_unique<Generator>(runtime_, r, open_opts(b.generator), model_, *variant);
        if (generator_->frames() != geom_.frames ||
            generator_->output_samples() != geom_.return_length * model_.upsample_factor) {
            throw UserError("generator geometry does not match the stream configuration");
        }
    }
    if (model_.index) {
        index_ = std::make_unique<FeatureIndex>(model_.index->dir);
        index_->set_nprobe(model_.index->nprobe);
        index_->reserve(content_->fixed_frames() + 1);
        XR_LOG_INFO("index: {} vectors, {} lists, nprobe {}", index_->ntotal(), index_->nlist(), index_->nprobe());
    }

    const int hf = content_->fixed_frames();
    scaled_in_.resize(static_cast<size_t>(hop_in_));
    resampled_in_.resize(in_resampler_.max_output(static_cast<size_t>(hop_in_)));
    window_.assign(static_cast<size_t>(window16k_), 0.0f);
    feats_.assign(static_cast<size_t>(hf + 1) * model_.feature_dim, 0.0f);
    feats_pre_.assign(feats_.size(), 0.0f);
    f0_cache_.assign(static_cast<size_t>(geom_.frames), 0.0f);
    voiced_cache_.assign(static_cast<size_t>(geom_.frames), 0.0f);
    if (rmvpe_) {
        f0_seg_.assign(static_cast<size_t>(rmvpe_->fixed_frames()), 0.0f);
        f0_tmp_.assign(f0_seg_.size(), 0.0f);
    }
    decoded_.assign(static_cast<size_t>(generator_->output_samples()), 0.0f);
    block_.assign(static_cast<size_t>(sola_.block()), 0.0f);
    out_scratch_.resize(out_resampler_.max_output(block_.size()));
    fifo_.assign(static_cast<size_t>(hop_out_) * 4 + out_scratch_.size() + 64, 0.0f);
    reset();

    XR_LOG_INFO("stream processor: {} | geometry {} | hop {} in / {} out samples | algorithmic latency {:.1f} ms",
                opts_.stream.to_string(), geom_.tag(), hop_in_, hop_out_, algorithmic_latency_seconds() * 1000.0);
}

double StreamProcessor::algorithmic_latency_seconds() const {
    // Emitted audio corresponds to window frames ending (C + S) frames before the newest
    // input; SOLA picks an offset in [0, S] (counted at its mean, S/2).
    const double frames = geom_.crossfade + geom_.sola_search - 0.5 * geom_.sola_search;
    return frames * 0.01 + in_resampler_.latency_seconds() + out_resampler_.latency_seconds();
}

void StreamProcessor::reset() {
    std::fill(window_.begin(), window_.end(), 0.0f);
    std::fill(f0_cache_.begin(), f0_cache_.end(), 0.0f);
    std::fill(voiced_cache_.begin(), voiced_cache_.end(), 0.0f);
    in_resampler_.reset();
    out_resampler_.reset();
    sola_.reset();
    // Prime the output FIFO with the output resampler's lookahead so every hop can be served.
    const size_t prime = out_resampler_.is_passthrough()
                             ? 0
                             : static_cast<size_t>(std::ceil(static_cast<double>(out_resampler_.lookahead_input()) *
                                                             opts_.output_rate / model_.sample_rate)) + 2;
    std::fill(fifo_.begin(), fifo_.end(), 0.0f);
    fifo_read_ = 0;
    fifo_size_ = prime;
}

std::vector<const SessionReport*> StreamProcessor::stage_reports() const {
    std::vector<const SessionReport*> r{&content_->session().report()};
    if (rmvpe_) r.push_back(&rmvpe_->session().report());
    r.push_back(&generator_->session().report());
    return r;
}

void StreamProcessor::run_pitch(const VoiceParamsSnapshot& p) {
    const std::span<const float> seg(window_.data() + (window16k_ - pitch_segment_), static_cast<size_t>(pitch_segment_));
    const auto f0 = rmvpe_->run_fixed(seg);
    const int n = static_cast<int>(f0.size());
    std::copy(f0.begin(), f0.end(), f0_seg_.begin());

    if (p.filter_radius >= 3) {
        const int k = p.filter_radius | 1;
        const int h = k / 2;
        float win[64];
        const int kk = std::min(k, 63);
        for (int i = 0; i < n; ++i) {
            int m = 0;
            for (int j = i - h; j <= i + h && m < kk; ++j) win[m++] = (j < 0 || j >= n) ? 0.0f : f0_seg_[static_cast<size_t>(j)];
            std::nth_element(win, win + m / 2, win + m);
            f0_tmp_[static_cast<size_t>(i)] = win[m / 2];
        }
        std::copy(f0_tmp_.begin(), f0_tmp_.begin() + n, f0_seg_.begin());
    }

    // Shift caches by one block, then write the newest n-4 frames (upstream rtrvc:
    // cache[4 - n:] = pitch[3:-1]).
    const int T = geom_.frames;
    const int H = geom_.block;
    std::memmove(f0_cache_.data(), f0_cache_.data() + H, static_cast<size_t>(T - H) * sizeof(float));
    std::memmove(voiced_cache_.data(), voiced_cache_.data() + H, static_cast<size_t>(T - H) * sizeof(float));
    const int m = std::min(n - 4, T);
    const int src0 = 3 + (n - 4 - m);
    for (int j = 0; j < m; ++j) {
        const float v = f0_seg_[static_cast<size_t>(src0 + j)];
        f0_cache_[static_cast<size_t>(T - m + j)] = v;
        voiced_cache_[static_cast<size_t>(T - m + j)] = v > 0.0f ? 1.0f : 0.0f;
    }
    if (p.fill_unvoiced_f0) {
        // np.interp over the voiced frames of the written segment (edges hold the nearest value).
        float* f = f0_cache_.data() + (T - m);
        int prev = -1;
        for (int j = 0; j <= m; ++j) {
            if (j < m && f[j] <= 0.0f) continue;
            if (j < m) {
                const int gap_start = prev + 1;
                for (int g = gap_start; g < j; ++g) {
                    f[g] = prev < 0 ? f[j] : f[prev] + (f[j] - f[prev]) * static_cast<float>(g - prev) / (j - prev);
                }
                prev = j;
            } else if (prev >= 0) {
                for (int g = prev + 1; g < m; ++g) f[g] = f[prev];
            }
        }
    }
}

void StreamProcessor::build_generator_inputs(const VoiceParamsSnapshot& p, std::span<const float> feats, int hf) {
    const int T = geom_.frames;
    const int D = model_.feature_dim;
    auto phone = generator_->phone();
    const bool protect = model_.uses_f0 && p.protect < 0.5f;
    for (int i = 0; i < T; ++i) {
        const int src = std::min(i / 2, hf);  // nearest x2 upsampling of (hf + 1) frames
        const float* a = feats.data() + static_cast<size_t>(src) * D;
        float* dst = phone.data() + static_cast<size_t>(i) * D;
        if (protect) {
            const float pf = voiced_cache_[static_cast<size_t>(i)] > 0.0f ? 1.0f : p.protect;
            const float* b = feats_pre_.data() + static_cast<size_t>(src) * D;
            for (int c = 0; c < D; ++c) dst[c] = a[c] * pf + b[c] * (1.0f - pf);
        } else {
            std::memcpy(dst, a, static_cast<size_t>(D) * sizeof(float));
        }
    }
    if (model_.uses_f0) {
        const float factor = std::pow(2.0f, p.pitch_shift / 12.0f);
        auto pitchf = generator_->pitchf();
        auto pitch = generator_->pitch();
        for (int i = 0; i < T; ++i) {
            const float f = f0_cache_[static_cast<size_t>(i)] * factor;
            pitchf[static_cast<size_t>(i)] = f;
            pitch[static_cast<size_t>(i)] = coarse_pitch(f);
        }
        for (auto& v : generator_->noise()) v = normal_(rng_);
    }
    for (auto& v : generator_->rnd()) v = normal_(rng_);
    generator_->set_speaker(std::clamp(p.speaker_id, 0, std::max(0, model_.n_speakers - 1)));
}

void StreamProcessor::process(std::span<const float> in, std::span<float> out, const VoiceParamsSnapshot& p,
                              HopTimings* timings) {
    const auto t0 = Clock::now();

    // 1. Gain and optional gate (per 10 ms frame, like upstream's threshold).
    const float gain = db_to_gain(p.input_gain_db);
    const size_t n_in = std::min(in.size(), scaled_in_.size());
    for (size_t i = 0; i < n_in; ++i) scaled_in_[i] = in[i] * gain;
    for (size_t i = n_in; i < scaled_in_.size(); ++i) scaled_in_[i] = 0.0f;
    if (p.silence_threshold_db > -60.0f) {
        const size_t zc = static_cast<size_t>(opts_.input_rate / 100);
        for (size_t s = 0; s + zc <= scaled_in_.size(); s += zc) {
            double e = 0.0;
            for (size_t i = s; i < s + zc; ++i) e += static_cast<double>(scaled_in_[i]) * scaled_in_[i];
            const double db = 10.0 * std::log10(e / static_cast<double>(zc) + 1e-12);
            if (db < p.silence_threshold_db) std::fill(scaled_in_.begin() + static_cast<std::ptrdiff_t>(s),
                                                       scaled_in_.begin() + static_cast<std::ptrdiff_t>(s + zc), 0.0f);
        }
    }

    // 2. Resample to 16 kHz and slide the analysis window.
    const size_t n16 = std::min(in_resampler_.process(scaled_in_, resampled_in_), window_.size());
    std::memmove(window_.data(), window_.data() + n16, (window_.size() - n16) * sizeof(float));
    std::memcpy(window_.data() + (window_.size() - n16), resampled_in_.data(), n16 * sizeof(float));
    const auto t1 = Clock::now();

    // 3. Content features for the whole window, plus a repeated last frame (upstream rtrvc).
    std::memcpy(content_->fixed_input().data(), window_.data(), window_.size() * sizeof(float));
    const auto fv = content_->run_fixed();
    const int hf = content_->fixed_frames();
    const int D = model_.feature_dim;
    std::memcpy(feats_.data(), fv.data(), fv.size() * sizeof(float));
    std::memcpy(feats_.data() + static_cast<size_t>(hf) * D, fv.data() + static_cast<size_t>(hf - 1) * D,
                static_cast<size_t>(D) * sizeof(float));
    const bool protect = model_.uses_f0 && p.protect < 0.5f;
    if (protect) std::memcpy(feats_pre_.data(), feats_.data(), feats_.size() * sizeof(float));
    const auto t2 = Clock::now();

    // 4. Pitch.
    if (rmvpe_) run_pitch(p);
    const auto t3 = Clock::now();

    // 5. Index retrieval on the frames that will be decoded (upstream: feats[skip_head // 2:]).
    if (index_ && p.index_rate > 0.0f) {
        const int start = std::min(geom_.skip_head / 2, hf);
        index_->blend(std::span<float>(feats_.data() + static_cast<size_t>(start) * D,
                                       static_cast<size_t>(hf + 1 - start) * D),
                      hf + 1 - start, std::min(p.index_rate, 1.0f));
    }
    const auto t4 = Clock::now();

    // 6. Generator.
    build_generator_inputs(p, feats_, hf);
    const auto audio = generator_->run();
    const auto t5 = Clock::now();

    // 7. Post: loudness envelope, SOLA crossfade, gain, output resampling.
    std::memcpy(decoded_.data(), audio.data(), audio.size() * sizeof(float));
    if (p.rms_mix_rate < 1.0f) {
        const size_t from = static_cast<size_t>(geom_.skip_head) * 160;
        const size_t len = static_cast<size_t>(geom_.return_length) * 160;
        rms_.apply(std::span<const float>(window_.data() + from, len), 160, decoded_, zc_model_,
                   std::max(p.rms_mix_rate, 0.0f));
    }
    const int offset = sola_.process(decoded_, block_);
    const float ogain = db_to_gain(p.output_gain_db);
    if (ogain != 1.0f) {
        for (auto& v : block_) v *= ogain;
    }
    const size_t produced = out_resampler_.process(block_, out_scratch_);
    // Append to FIFO (compact first if needed).
    if (fifo_read_ + fifo_size_ + produced > fifo_.size()) {
        std::memmove(fifo_.data(), fifo_.data() + fifo_read_, fifo_size_ * sizeof(float));
        fifo_read_ = 0;
    }
    const size_t room = fifo_.size() - (fifo_read_ + fifo_size_);
    const size_t push = std::min(produced, room);
    std::memcpy(fifo_.data() + fifo_read_ + fifo_size_, out_scratch_.data(), push * sizeof(float));
    fifo_size_ += push;
    const size_t pop = std::min(out.size(), fifo_size_);
    std::memcpy(out.data(), fifo_.data() + fifo_read_, pop * sizeof(float));
    for (size_t i = pop; i < out.size(); ++i) out[i] = 0.0f;
    fifo_read_ += pop;
    fifo_size_ -= pop;
    const auto t6 = Clock::now();

    if (timings) {
        timings->resample_in = ms_between(t0, t1);
        timings->content = ms_between(t1, t2);
        timings->pitch = ms_between(t2, t3);
        timings->index = ms_between(t3, t4);
        timings->generator = ms_between(t4, t5);
        timings->post = ms_between(t5, t6);
        timings->total = ms_between(t0, t6);
        timings->sola_offset = offset;
    }
}

}  // namespace xr::rvc
