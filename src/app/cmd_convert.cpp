// convert / benchmark / profile: everything that drives the streaming processor offline.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

#include <spdlog/fmt/fmt.h>
#include <nlohmann/json.hpp>

#include "app/cli_context.h"
#include "app/pipeline_cli.h"
#include "audio/wav_io.h"
#include "dsp/resampler.h"
#include "rvc/model_info.h"
#include "rvc/offline.h"
#include "util/error.h"

namespace xr::cli {

namespace {

nlohmann::json stats_json(const StatsSummary& s) {
    return {{"mean", s.mean}, {"median", s.median}, {"p95", s.p95}, {"max", s.max}};
}

nlohmann::json stage_json(const rvc::StreamProcessor& p) {
    auto arr = nlohmann::json::array();
    for (const SessionReport* r : p.stage_reports()) {
        nlohmann::json j;
        to_json(j, *r);
        arr.push_back(j);
    }
    return arr;
}

void print_timing_table(const rvc::OfflineResult& r, double hop_ms) {
    auto row = [](const char* name, const StatsSummary& s) {
        std::printf("  %-14s %8.2f %8.2f %8.2f %8.2f\n", name, s.mean, s.median, s.p95, s.max);
    };
    std::printf("Timing per hop (ms)        mean   median      p95      max\n");
    row("resample in", r.resample_in);
    row("content enc.", r.content);
    row("pitch (RMVPE)", r.pitch);
    row("index", r.index);
    row("generator", r.generator);
    row("postprocess", r.post);
    row("TOTAL", r.total);
    std::printf("  hop budget %.0f ms; p95 uses %.0f%% of it; realtime factor %.3f (%.2f s audio in %.2f s)\n", hop_ms,
                100.0 * r.total.p95 / hop_ms, r.realtime_factor(), r.audio_seconds, r.wall_seconds);
}

// Synthetic voiced test signal used when no input file is given: a harmonic source with
// a 110-220 Hz F0 contour, formant-like spectral tilt and pauses. Not speech, but it
// exercises every stage with realistic shapes. Labelled as synthetic in all output.
std::vector<float> synthetic_voice(int rate, double seconds) {
    std::vector<float> x(static_cast<size_t>(rate * seconds));
    double phase = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double t = static_cast<double>(i) / rate;
        const double f0 = 165.0 + 55.0 * std::sin(2 * 3.14159265 * 0.3 * t);
        phase += 2 * 3.14159265 * f0 / rate;
        double v = 0.0;
        for (int h = 1; h <= 30; ++h) v += std::sin(h * phase) / (h * std::sqrt(static_cast<double>(h)));
        const double env = std::fmod(t, 2.0) < 1.6 ? 0.25 : 0.0;
        x[i] = static_cast<float>(env * v * 0.3);
    }
    return x;
}

struct ConvertArgs {
    PipelineArgs pipe;
    std::string input, output;
    int output_rate = 0;
    std::string report;
    bool pcm16 = false;
};

int run_convert(const GlobalOptions& global, const ConvertArgs& a) {
    CliContext ctx(global);
    const auto model = rvc::load_model_info(a.pipe.model);
    XR_LOG_INFO("model: {}", model.describe());
    const AudioBuffer wav = read_audio_file(a.input);
    std::vector<float> in = wav.samples;
    int in_rate = wav.sample_rate;
    if (in_rate % 100 != 0) {  // e.g. 22050 Hz files: bring to 48 kHz first
        in = dsp::Resampler::resample(in, in_rate, 48000);
        in_rate = 48000;
    }
    rvc::ProcessorOptions opts;
    opts.stream = a.pipe.stream();
    opts.input_rate = in_rate;
    opts.output_rate = a.output_rate ? a.output_rate : model.sample_rate;
    opts.backends = a.pipe.backends();
    opts.seed = a.pipe.seed;
    opts.resampler_quality = a.pipe.resampler_quality();

    const auto t_load = std::chrono::steady_clock::now();
    rvc::StreamProcessor proc(ctx.runtime(), model, opts);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
    print_stage_reports(proc);

    const auto result = rvc::convert_offline(proc, in, a.pipe.params(), [](double f) {
        std::fprintf(stderr, "\rconverting... %3.0f%%", f * 100.0);
    });
    std::fprintf(stderr, "\n");
    write_wav(a.output, result.audio, result.sample_rate,
              a.pcm16 ? WavSampleFormat::Int16 : WavSampleFormat::Float32);
    std::printf("\nwrote %s (%.2f s, %d Hz); models loaded in %.1f s\n", a.output.c_str(),
                static_cast<double>(result.audio.size()) / result.sample_rate, result.sample_rate, load_s);
    print_timing_table(result, proc.hop_seconds() * 1000.0);
    std::printf("  algorithmic latency %.1f ms (compensated in the output file)\n",
                proc.algorithmic_latency_seconds() * 1000.0);
    if (!a.report.empty()) {
        nlohmann::json j{{"input", a.input}, {"output", a.output}, {"model", model.name},
                         {"stream", opts.stream.to_string()}, {"stages", stage_json(proc)},
                         {"realtime_factor", result.realtime_factor()},
                         {"timing_ms", {{"content_encoder", stats_json(result.content)},
                                        {"pitch", stats_json(result.pitch)},
                                        {"index", stats_json(result.index)},
                                        {"generator", stats_json(result.generator)},
                                        {"postprocess", stats_json(result.post)},
                                        {"total", stats_json(result.total)}}}};
        std::ofstream(a.report) << j.dump(2);
    }
    return 0;
}

struct BenchArgs {
    PipelineArgs pipe;
    std::string input;
    double seconds = 10.0;
    std::string json_out;
    bool sweep = false;
};

nlohmann::json bench_one(CliContext& ctx, const rvc::ModelInfo& model, const PipelineArgs& pipe,
                         const rvc::StreamConfig& stream, const std::vector<float>& audio, int rate, bool synthetic,
                         bool print) {
    rvc::ProcessorOptions opts;
    opts.stream = stream;
    opts.input_rate = rate;
    opts.output_rate = 48000;
    opts.backends = pipe.backends();
    opts.seed = 1;
    opts.resampler_quality = pipe.resampler_quality();
    rvc::StreamProcessor proc(ctx.runtime(), model, opts);
    const auto params = pipe.params();
    // Warm-up pass (first runs allocate arenas / JIT kernels), then the measured pass.
    {
        std::vector<float> warm(audio.begin(), audio.begin() + std::min<size_t>(audio.size(), rate * 2));
        rvc::convert_offline(proc, warm, params);
    }
    const auto r = rvc::convert_offline(proc, audio, params);
    const double hop_ms = proc.hop_seconds() * 1000.0;
    if (print) {
        std::printf("\n== %s (%s) ==\n", stream.to_string().c_str(), proc.geometry().tag().c_str());
        print_stage_reports(proc);
        print_timing_table(r, hop_ms);
        std::printf("  algorithmic latency %.1f ms + one hop of buffering (%.0f ms) + processing + device buffers\n",
                    proc.algorithmic_latency_seconds() * 1000.0, hop_ms);
    }
    std::string backend_summary;
    bool npu = false;
    for (const SessionReport* s : proc.stage_reports()) {
        backend_summary += (backend_summary.empty() ? "" : ",") + s->stage + "=" + std::string(to_string(s->effective));
        npu = npu || s->npu_evidence();
    }
    return {{"backend", pipe.backend},
            {"effective_backends", backend_summary},
            {"npu_evidence", npu},
            {"input", synthetic ? "synthetic" : "file"},
            {"stream", {{"block_ms", stream.block_ms}, {"crossfade_ms", stream.crossfade_ms}, {"extra_ms", stream.extra_ms}}},
            {"hop_ms", hop_ms},
            {"content_encoder_ms", r.content.mean},
            {"pitch_ms", r.pitch.mean},
            {"index_ms", r.index.mean},
            {"generator_ms", r.generator.mean},
            {"postprocess_ms", r.post.mean},
            {"total_inference_ms", r.total.mean},
            {"total_inference_p95_ms", r.total.p95},
            {"total_inference_max_ms", r.total.max},
            {"realtime_factor", r.realtime_factor()},
            {"algorithmic_latency_ms", proc.algorithmic_latency_seconds() * 1000.0},
            {"hops", r.hops},
            {"stats_ms", {{"content_encoder", stats_json(r.content)}, {"pitch", stats_json(r.pitch)},
                          {"index", stats_json(r.index)}, {"generator", stats_json(r.generator)},
                          {"postprocess", stats_json(r.post)}, {"total", stats_json(r.total)}}},
            {"stages", stage_json(proc)}};
}

int run_benchmark(const GlobalOptions& global, const BenchArgs& a) {
    CliContext ctx(global);
    const auto model = rvc::load_model_info(a.pipe.model);
    std::vector<float> audio;
    int rate = 48000;
    bool synthetic = a.input.empty();
    if (synthetic) {
        audio = synthetic_voice(rate, a.seconds);
        std::printf("input: synthetic voiced signal, %.1f s (pass --input to benchmark real speech)\n", a.seconds);
    } else {
        const AudioBuffer wav = read_audio_file(a.input);
        audio = dsp::Resampler::resample(wav.samples, wav.sample_rate, rate);
    }
    nlohmann::json results = nlohmann::json::array();
    std::vector<rvc::StreamConfig> streams;
    if (a.sweep) {
        for (const auto& g : model.generators) streams.push_back(g.stream);
    } else {
        streams.push_back(a.pipe.stream());
    }
    for (const auto& s : streams) results.push_back(bench_one(ctx, model, a.pipe, s, audio, rate, synthetic, !global.json));
    const nlohmann::json out = a.sweep ? nlohmann::json{{"model", model.name}, {"results", results}} : results[0];
    if (global.json) std::printf("%s\n", out.dump(2).c_str());
    if (!a.json_out.empty()) {
        std::ofstream(a.json_out) << out.dump(2);
        std::printf("\nbenchmark JSON written to %s\n", a.json_out.c_str());
    }
    return 0;
}

struct ProfileArgs {
    PipelineArgs pipe;
    std::string input;
    double seconds = 5.0;
};

int run_profile(const GlobalOptions& global, const ProfileArgs& a) {
    CliContext ctx(global);
    const auto model = rvc::load_model_info(a.pipe.model);
    rvc::ProcessorOptions opts;
    opts.stream = a.pipe.stream();
    opts.input_rate = 48000;
    opts.output_rate = 48000;
    opts.backends = a.pipe.backends();
    opts.backends.keep_profiling = true;
    opts.backends.ai_analyzer = true;
    opts.seed = 1;
    rvc::StreamProcessor proc(ctx.runtime(), model, opts);
    std::vector<float> audio;
    if (a.input.empty()) {
        audio = synthetic_voice(48000, a.seconds);
    } else {
        const AudioBuffer wav = read_audio_file(a.input);
        audio = dsp::Resampler::resample(wav.samples, wav.sample_rate, 48000);
    }
    const auto r = rvc::convert_offline(proc, audio, a.pipe.params());
    std::printf("Profiled %d hops (%s)\n\n", r.hops, opts.stream.to_string().c_str());
    print_stage_reports(proc);
    std::printf("\nORT profiles (session-long) are written when the process exits, to %s/.\n"
                "Inspect them with: python tools/inspect_execution.py %s\n",
                opts.backends.diagnostics_dir.string().c_str(), opts.backends.diagnostics_dir.string().c_str());
    print_timing_table(r, proc.hop_seconds() * 1000.0);
    return 0;
}

}  // namespace

void register_convert(CLI::App& app, GlobalOptions& global, int& exit_code) {
    auto a = std::make_shared<ConvertArgs>();
    auto* cmd = app.add_subcommand("convert", "Convert a WAV file through the realtime streaming pipeline");
    cmd->add_option("input,--input,-i", a->input, "Input audio file")->required()->check(CLI::ExistingFile);
    cmd->add_option("output,--output,-o", a->output, "Output WAV file")->required();
    cmd->add_option("--output-rate", a->output_rate, "Output sample rate (default: model rate)");
    cmd->add_option("--report", a->report, "Write a JSON report");
    cmd->add_flag("--pcm16", a->pcm16, "Write 16-bit PCM instead of float");
    a->pipe.add_to(*cmd, false);
    cmd->callback([&global, &exit_code, a] { exit_code = run_convert(global, *a); });

    auto b = std::make_shared<BenchArgs>();
    auto* bench = app.add_subcommand("benchmark", "Benchmark the pipeline faster than realtime (no audio device)");
    b->pipe.add_to(*bench, true);
    bench->add_option("--input,-i", b->input, "Speech file to benchmark with (default: synthetic signal)");
    bench->add_option("--seconds", b->seconds, "Length of the synthetic signal")->capture_default_str();
    bench->add_option("--json-out", b->json_out, "Write results as JSON");
    bench->add_flag("--sweep", b->sweep, "Benchmark every stream configuration exported for the model");
    bench->callback([&global, &exit_code, b] { exit_code = run_benchmark(global, *b); });

    auto p = std::make_shared<ProfileArgs>();
    auto* prof = app.add_subcommand("profile", "Run with ORT profiling and report per-stage provider placement");
    p->pipe.add_to(*prof, true);
    prof->add_option("--input,-i", p->input, "Audio file (default: synthetic signal)");
    prof->callback([&global, &exit_code, p] { exit_code = run_profile(global, *p); });
}

}  // namespace xr::cli
