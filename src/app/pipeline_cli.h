#pragma once

#include <string>

#include <CLI/CLI.hpp>

#include "rvc/stream_processor.h"
#include "rvc/voice_params.h"

namespace xr::cli {

// Command line options shared by convert / benchmark / profile.
struct PipelineArgs {
    std::string model;
    std::string backend = "auto";
    std::string content_backend, pitch_backend, generator_backend;
    bool no_fallback = false;
    std::string preset = "balanced";
    int block_ms = 0, crossfade_ms = 0, extra_ms = 0, lookahead_ms = -1;
    int threads = 0;
    std::string cache_dir = "cache";
    std::string diagnostics_dir = "diagnostics";
    unsigned long long seed = 0;
    std::string quality = "balanced";

    float pitch = 0.0f;
    float index_rate = 0.5f;
    float rms_mix = 0.25f;
    float protect = 0.33f;
    int speaker = 0;
    int filter_radius = 0;
    float input_gain_db = 0.0f;
    float output_gain_db = 0.0f;
    float gate_db = -60.0f;
    bool no_f0_fill = false;

    void add_to(CLI::App& cmd, bool model_positional);
    rvc::StreamConfig stream() const;
    rvc::StageBackends backends() const;
    rvc::VoiceParamsSnapshot params() const;
    dsp::ResamplerQuality resampler_quality() const;
};

// Prints one line per neural stage: requested/effective backend, fallbacks, evidence.
void print_stage_reports(const rvc::StreamProcessor& p);

}  // namespace xr::cli
