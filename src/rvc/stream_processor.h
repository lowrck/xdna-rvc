#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "dsp/resampler.h"
#include "dsp/sola.h"
#include "inference/backend_kind.h"
#include "inference/session_report.h"
#include "rvc/content_encoder.h"
#include "rvc/feature_index.h"
#include "rvc/generator.h"
#include "rvc/model_info.h"
#include "rvc/rmvpe.h"
#include "rvc/stream_config.h"
#include "rvc/voice_params.h"

namespace xr::rvc {

// Backend choice per neural stage. Different stages may use different backends,
// e.g. content encoder on XDNA 2, RMVPE on CPU, generator on XDNA 2.
struct StageBackends {
    BackendKind content = BackendKind::Auto;
    BackendKind pitch = BackendKind::Auto;
    BackendKind generator = BackendKind::Auto;
    bool allow_cpu_fallback = true;  // fallbacks are always logged and reported
    int intra_op_threads = 0;        // 0 = ORT default
    bool collect_evidence = true;
    bool keep_profiling = false;     // session-long ORT profiles (diagnostics)
    bool ai_analyzer = false;
    std::filesystem::path cache_dir = "cache";
    std::filesystem::path diagnostics_dir = "diagnostics";
};

struct ProcessorOptions {
    StreamConfig stream;
    int input_rate = 48000;
    int output_rate = 48000;
    StageBackends backends;
    dsp::ResamplerQuality resampler_quality = dsp::ResamplerQuality::Balanced;
    uint64_t seed = 0;  // 0 = random
};

// Per-hop timing breakdown, milliseconds.
struct HopTimings {
    double resample_in = 0, content = 0, pitch = 0, index = 0, generator = 0, post = 0, total = 0;
    int sola_offset = 0;
};

// The streaming RVC voice conversion pipeline. One call to process() consumes one
// hop of device-rate input and produces one hop of device-rate output:
//
//   input -> gain/gate -> resample 16 kHz -> sliding window (T frames)
//   window -> content encoder -> (+1 frame) -> index blend (frames >= E/2) -> x2 repeat
//   newest segment -> RMVPE -> F0 cache (shift by H, write newest frames)
//   features + F0 -> generator (decodes frames [E, E+R)) -> RMS mix -> SOLA -> gain
//   -> resample to output rate -> FIFO -> output hop
//
// The same object is used for realtime audio (from the inference worker thread),
// offline WAV conversion and benchmarking, so all three exercise identical code.
// process() is allocation-free after construction.
class StreamProcessor {
public:
    StreamProcessor(OrtRuntime& runtime, const ModelInfo& model, const ProcessorOptions& options);

    const ModelInfo& model() const { return model_; }
    const StreamGeometry& geometry() const { return geom_; }
    const ProcessorOptions& options() const { return opts_; }
    int hop_input_samples() const { return hop_in_; }
    int hop_output_samples() const { return hop_out_; }
    double hop_seconds() const { return geom_.block * 0.01; }

    // Delay from input to output introduced by the algorithm (right context, SOLA
    // centering, resampler filters). Excludes processing time and device buffers.
    double algorithmic_latency_seconds() const;

    void process(std::span<const float> in, std::span<float> out, const VoiceParamsSnapshot& params,
                 HopTimings* timings = nullptr);
    void reset();

    // Execution provenance of each neural stage.
    std::vector<const SessionReport*> stage_reports() const;
    bool has_index() const { return index_ != nullptr; }

private:
    void run_pitch(const VoiceParamsSnapshot& p);
    void build_generator_inputs(const VoiceParamsSnapshot& p, std::span<const float> feats, int hubert_frames);

    OrtRuntime& runtime_;
    ModelInfo model_;
    ProcessorOptions opts_;
    StreamGeometry geom_;
    int hop_in_ = 0, hop_out_ = 0;
    int window16k_ = 0;     // T * 160
    int pitch_segment_ = 0; // RMVPE segment length (16 kHz samples)
    int zc_model_ = 0;      // model samples per 10 ms frame

    std::unique_ptr<ContentEncoder> content_;
    std::unique_ptr<RmvpePitch> rmvpe_;
    std::unique_ptr<FeatureIndex> index_;
    std::unique_ptr<Generator> generator_;

    dsp::Resampler in_resampler_;
    dsp::Resampler out_resampler_;
    dsp::Sola sola_;
    dsp::RmsMixer rms_;

    std::vector<float> scaled_in_;      // gain/gate applied input hop
    std::vector<float> resampled_in_;   // resampler output scratch
    std::vector<float> window_;         // 16 kHz sliding window, T*160 samples
    std::vector<float> feats_;          // (hubert frames + 1) x dim, after index blending
    std::vector<float> feats_pre_;      // same, before blending (for protect)
    std::vector<float> f0_cache_;       // T frames, raw F0 (Hz, unshifted), filled per settings
    std::vector<float> voiced_cache_;   // T frames, 1 = voiced by RMVPE
    std::vector<float> f0_seg_;         // scratch: newest F0 segment
    std::vector<float> f0_tmp_;
    std::vector<float> decoded_;        // generator output after RMS mix
    std::vector<float> block_;          // SOLA output, model rate
    std::vector<float> out_scratch_;    // output resampler scratch
    std::vector<float> fifo_;           // output FIFO (device rate)
    size_t fifo_read_ = 0, fifo_size_ = 0;
    std::mt19937 rng_;
    std::normal_distribution<float> normal_{0.0f, 1.0f};
};

}  // namespace xr::rvc
