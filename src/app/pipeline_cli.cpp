#include "app/pipeline_cli.h"

#include <cstdio>

namespace xr::cli {

void PipelineArgs::add_to(CLI::App& cmd, bool model_positional) {
    if (model_positional) {
        cmd.add_option("model", model, "Voice model.json")->required()->check(CLI::ExistingFile);
    } else {
        cmd.add_option("--model,-m", model, "Voice model.json")->required()->check(CLI::ExistingFile);
    }
    cmd.add_option("--backend,-b", backend, "Backend for all neural stages: auto|xdna2|directml|cpu")
        ->capture_default_str();
    cmd.add_option("--content-backend", content_backend, "Override backend for the content encoder");
    cmd.add_option("--pitch-backend", pitch_backend, "Override backend for RMVPE");
    cmd.add_option("--generator-backend", generator_backend, "Override backend for the RVC generator");
    cmd.add_flag("--no-fallback", no_fallback, "Fail instead of falling back to CPU when a backend fails");
    cmd.add_option("--preset", preset, "Stream preset: low_latency|balanced|quality")->capture_default_str();
    cmd.add_option("--block-ms", block_ms, "Hop size (overrides preset)");
    cmd.add_option("--crossfade-ms", crossfade_ms, "Crossfade length (overrides preset)");
    cmd.add_option("--extra-ms", extra_ms, "Left context (overrides preset)");
    cmd.add_option("--lookahead-ms", lookahead_ms, "Right context beyond the decoded region (overrides preset)");
    cmd.add_option("--cache-dir", cache_dir, "Compiled-model / static-shape cache directory")->capture_default_str();
    cmd.add_option("--diagnostics-dir", diagnostics_dir, "ORT profiles and reports")->capture_default_str();
    cmd.add_option("--seed", seed, "Noise seed (0 = random) for reproducible output");
    cmd.add_option("--resampler", quality, "Resampler quality: fast|balanced|high")->capture_default_str();

    cmd.add_option("--pitch,-p", pitch, "Pitch shift in semitones")->capture_default_str();
    cmd.add_option("--index-rate", index_rate, "Index feature blend 0..1")->capture_default_str();
    cmd.add_option("--rms-mix", rms_mix, "Loudness envelope mix 0..1 (1 = model loudness)")->capture_default_str();
    cmd.add_option("--protect", protect, "Consonant protection 0..0.5 (0.5 = off)")->capture_default_str();
    cmd.add_option("--speaker", speaker, "Speaker id")->capture_default_str();
    cmd.add_option("--filter-radius", filter_radius, "F0 median filter length (>= 3 enables)");
    cmd.add_option("--input-gain", input_gain_db, "Input gain dB");
    cmd.add_option("--output-gain", output_gain_db, "Output gain dB");
    cmd.add_option("--gate", gate_db, "Silence gate threshold dB (<= -60 disables)");
    cmd.add_flag("--no-f0-fill", no_f0_fill, "Keep F0 at zero in unvoiced frames instead of interpolating");
}

rvc::StreamConfig PipelineArgs::stream() const {
    rvc::StreamConfig s = rvc::preset(preset);
    if (block_ms) s.block_ms = block_ms;
    if (crossfade_ms) s.crossfade_ms = crossfade_ms;
    if (extra_ms) s.extra_ms = extra_ms;
    if (lookahead_ms >= 0) s.lookahead_ms = lookahead_ms;
    s.validate();
    return s;
}

rvc::StageBackends PipelineArgs::backends() const {
    rvc::StageBackends b;
    const BackendKind all = parse_backend(backend);
    b.content = content_backend.empty() ? all : parse_backend(content_backend);
    b.pitch = pitch_backend.empty() ? all : parse_backend(pitch_backend);
    b.generator = generator_backend.empty() ? all : parse_backend(generator_backend);
    b.allow_cpu_fallback = !no_fallback;
    b.intra_op_threads = threads;
    b.cache_dir = cache_dir;
    b.diagnostics_dir = diagnostics_dir;
    return b;
}

rvc::VoiceParamsSnapshot PipelineArgs::params() const {
    rvc::VoiceParamsSnapshot p;
    p.pitch_shift = pitch;
    p.index_rate = index_rate;
    p.rms_mix_rate = rms_mix;
    p.protect = protect;
    p.speaker_id = speaker;
    p.filter_radius = filter_radius;
    p.input_gain_db = input_gain_db;
    p.output_gain_db = output_gain_db;
    p.silence_threshold_db = gate_db;
    p.fill_unvoiced_f0 = !no_f0_fill;
    return p;
}

dsp::ResamplerQuality PipelineArgs::resampler_quality() const {
    if (quality == "fast") return dsp::ResamplerQuality::Fast;
    if (quality == "high") return dsp::ResamplerQuality::High;
    return dsp::ResamplerQuality::Balanced;
}

void print_stage_reports(const rvc::StreamProcessor& p) {
    std::printf("Execution\n---------\n");
    for (const SessionReport* r : p.stage_reports()) {
        std::printf("  %-16s %-9s", r->stage.c_str(), std::string(display_name(r->effective)).c_str());
        if (r->fell_back) std::printf(" FALLBACK from %s", std::string(display_name(r->requested)).c_str());
        if (r->npu_evidence()) std::printf(" [NPU nodes confirmed]");
        std::printf("\n");
        for (const auto& u : r->provider_usage) {
            const double frac = r->time_fraction(u.provider).value_or(0.0) * 100.0;
            std::printf("      %-28s %4d nodes  %5.1f%% of node time\n", u.provider.c_str(), u.node_count, frac);
        }
        if (r->vitisai_report) {
            std::printf("      VitisAI report: %d nodes, NPU %d, CPU %d\n", r->vitisai_report->nodes_total(),
                        r->vitisai_report->nodes_on("NPU"), r->vitisai_report->nodes_on("CPU"));
        }
        if (r->fell_back) std::printf("      reason: %s\n", r->fallback_reason.c_str());
        if (!r->evidence_error.empty()) std::printf("      evidence: %s\n", r->evidence_error.c_str());
    }
    std::printf("  %-16s %-9s %s\n", "index", "CPU", p.has_index() ? "(IVF-Flat)" : "(no index)");
}

}  // namespace xr::cli
