// `xdna-rvc-cli features`: WAV -> HuBERT/ContentVec features (.npy), for validating
// the native content-encoder path against the Python reference.

#include <chrono>
#include <cstdio>

#include <fmt/format.h>

#include "app/cli_context.h"
#include "audio/wav_io.h"
#include "dsp/resampler.h"
#include "rvc/content_encoder.h"
#include "util/npy.h"

namespace xr::cli {

namespace {

struct FeaturesArgs {
    std::string input;
    std::string encoder;
    std::string backend = "cpu";
    std::string out;
    std::string dump_audio;
    int repeat = 1;
    bool allow_fallback = false;
};

int run_features(const GlobalOptions& global, const FeaturesArgs& a) {
    CliContext ctx(global);
    const AudioBuffer wav = read_audio_file(a.input);
    XR_LOG_INFO("input {}: {:.2f} s at {} Hz ({} channel(s))", a.input, wav.duration_seconds(), wav.sample_rate,
                wav.source_channels);
    const auto audio16k = dsp::Resampler::resample(wav.samples, wav.sample_rate, rvc::ContentEncoder::kSampleRate);

    SessionRequest req;
    req.stage = "content_encoder";
    req.model_path = a.encoder;
    OpenOptions opts;
    opts.backend = parse_backend(a.backend);
    opts.allow_cpu_fallback = a.allow_fallback;
    rvc::ContentEncoder enc(ctx.runtime(), req, opts);

    int frames = 0;
    std::vector<float> feats;
    double best_ms = 1e30, total_ms = 0;
    for (int i = 0; i < std::max(1, a.repeat); ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        feats = enc.extract(audio16k, frames);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        best_ms = std::min(best_ms, ms);
        total_ms += ms;
    }
    write_npy(a.out, feats, {frames, enc.feature_dim()});
    if (!a.dump_audio.empty()) {
        write_npy(a.dump_audio, audio16k, {static_cast<int64_t>(audio16k.size())});
    }
    const double audio_ms = 1000.0 * static_cast<double>(audio16k.size()) / rvc::ContentEncoder::kSampleRate;
    std::printf("content encoder: %s\n", enc.session().report().summary().c_str());
    std::printf("features: %d frames x %d dims -> %s\n", frames, enc.feature_dim(), a.out.c_str());
    std::printf("timing: best %.1f ms, mean %.1f ms for %.0f ms of audio (realtime factor %.3f)\n", best_ms,
                total_ms / std::max(1, a.repeat), audio_ms, best_ms / audio_ms);
    return 0;
}

}  // namespace

void register_features(CLI::App& app, GlobalOptions& global, int& exit_code) {
    auto args = std::make_shared<FeaturesArgs>();
    auto* cmd = app.add_subcommand("features", "Extract HuBERT/ContentVec features from an audio file to .npy");
    cmd->add_option("input", args->input, "Input audio file (WAV/FLAC/MP3)")->required()->check(CLI::ExistingFile);
    cmd->add_option("--encoder", args->encoder, "Content encoder ONNX (content_encoder_v1/v2.onnx)")
        ->required()
        ->check(CLI::ExistingFile);
    cmd->add_option("--out,-o", args->out, "Output .npy [frames, dim]")->required();
    cmd->add_option("--backend", args->backend, "auto|xdna2|directml|cpu")->capture_default_str();
    cmd->add_flag("--allow-cpu-fallback", args->allow_fallback, "Fall back to CPU if the backend fails (logged)");
    cmd->add_option("--dump-audio", args->dump_audio, "Also write the 16 kHz model input as .npy");
    cmd->add_option("--repeat", args->repeat, "Run N times and report timing")->capture_default_str();
    cmd->callback([&global, &exit_code, args] { exit_code = run_features(global, *args); });
}

}  // namespace xr::cli
