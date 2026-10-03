// inspect-model and f0 (pitch debugging).

#include <cstdio>

#include <nlohmann/json.hpp>

#include "app/cli_context.h"
#include "audio/wav_io.h"
#include "dsp/resampler.h"
#include "rvc/model_info.h"
#include "rvc/rmvpe.h"
#include "util/error.h"
#include "util/npy.h"

namespace xr::cli {

namespace {

int run_inspect(const GlobalOptions& global, const std::string& path) {
    const auto m = rvc::load_model_info(path);
    if (global.json) {
        nlohmann::json gens = nlohmann::json::array();
        for (const auto& g : m.generators) {
            gens.push_back({{"path", g.path.string()}, {"block_ms", g.stream.block_ms},
                            {"crossfade_ms", g.stream.crossfade_ms}, {"extra_ms", g.stream.extra_ms},
                            {"frames", g.frames}, {"skip_head", g.skip_head}, {"return_length", g.return_length},
                            {"precision", g.precision}, {"validated", g.validated},
                            {"exists", std::filesystem::exists(g.path)}});
        }
        nlohmann::json j{{"name", m.name}, {"rvc_version", m.rvc_version}, {"feature_dim", m.feature_dim},
                         {"uses_f0", m.uses_f0}, {"sample_rate", m.sample_rate}, {"n_speakers", m.n_speakers},
                         {"source_file", m.source_file}, {"source_sha256", m.source_sha256},
                         {"conversion_version", m.conversion_version},
                         {"content_encoder", m.content_encoder.string()}, {"rmvpe", m.rmvpe.string()},
                         {"generators", gens}, {"warnings", m.warnings}};
        if (m.index) j["index"] = {{"dir", m.index->dir.string()}, {"ntotal", m.index->ntotal}, {"nlist", m.index->nlist}};
        std::printf("%s\n", j.dump(2).c_str());
        return 0;
    }
    auto exists = [](const std::filesystem::path& p) { return std::filesystem::exists(p) ? "ok" : "MISSING"; };
    std::printf("Model\n-----\n");
    std::printf("  Name             : %s\n", m.name.c_str());
    std::printf("  Source           : %s (sha256 %.16s...)\n", m.source_file.c_str(), m.source_sha256.c_str());
    std::printf("  RVC version      : %s\n", m.rvc_version.c_str());
    std::printf("  Feature dim      : %d\n", m.feature_dim);
    std::printf("  F0               : %s\n", m.uses_f0 ? "yes" : "no");
    std::printf("  Output rate      : %d Hz\n", m.sample_rate);
    std::printf("  Speakers         : %d\n", m.n_speakers);
    std::printf("  Conversion       : %s\n", m.conversion_version.c_str());
    std::printf("  Content encoder  : %s [%s]\n", m.content_encoder.string().c_str(), exists(m.content_encoder));
    if (m.uses_f0) std::printf("  RMVPE            : %s [%s]\n", m.rmvpe.string().c_str(), exists(m.rmvpe));
    std::printf("  Index            : %s\n",
                m.index ? (std::to_string(m.index->ntotal) + " vectors, IVF" + std::to_string(m.index->nlist)).c_str()
                        : "none");
    std::printf("\nGenerator variants (fixed shapes)\n");
    for (const auto& g : m.generators) {
        std::printf("  block %3d ms, crossfade %3d ms, extra %4d ms -> T=%d skip=%d return=%d  %s  %s [%s]\n",
                    g.stream.block_ms, g.stream.crossfade_ms, g.stream.extra_ms, g.frames, g.skip_head,
                    g.return_length, g.precision.c_str(), g.validated ? "validated" : "VALIDATION FAILED",
                    exists(g.path));
    }
    for (const auto& w : m.warnings) std::printf("warning: %s\n", w.c_str());
    return 0;
}

struct F0Args {
    std::string input, model, out, backend = "cpu", dump_audio;
};

int run_f0(const GlobalOptions& global, const F0Args& a) {
    CliContext ctx(global);
    const auto m = rvc::load_model_info(a.model);
    if (!m.uses_f0) throw UserError("model " + m.name + " does not use F0");
    const AudioBuffer wav = read_audio_file(a.input);
    const auto audio16k = dsp::Resampler::resample(wav.samples, wav.sample_rate, 16000);
    SessionRequest req;
    req.model_path = m.rmvpe;
    OpenOptions opts;
    opts.backend = parse_backend(a.backend);
    rvc::RmvpePitch rmvpe(ctx.runtime(), req, opts, m.rmvpe_mel_basis);
    const auto f0 = rmvpe.extract(audio16k);
    write_npy(a.out, f0, {static_cast<int64_t>(f0.size())});
    if (!a.dump_audio.empty()) write_npy(a.dump_audio, audio16k, {static_cast<int64_t>(audio16k.size())});
    int voiced = 0;
    for (float v : f0) voiced += v > 0;
    std::printf("%zu frames, %d voiced -> %s\n", f0.size(), voiced, a.out.c_str());
    return 0;
}

}  // namespace

void register_inspect(CLI::App& app, GlobalOptions& global, int& exit_code) {
    auto path = std::make_shared<std::string>();
    auto* cmd = app.add_subcommand("inspect-model", "Show a converted voice model (model.json)");
    cmd->add_option("model", *path, "model.json")->required()->check(CLI::ExistingFile);
    cmd->callback([&global, &exit_code, path] { exit_code = run_inspect(global, *path); });

    auto a = std::make_shared<F0Args>();
    auto* f0 = app.add_subcommand("f0", "Extract RMVPE pitch from an audio file to .npy (debugging)");
    f0->add_option("input", a->input, "Audio file")->required()->check(CLI::ExistingFile);
    f0->add_option("--model,-m", a->model, "Voice model.json (for the RMVPE path)")->required();
    f0->add_option("--out,-o", a->out, "Output .npy (Hz per 10 ms frame)")->required();
    f0->add_option("--backend", a->backend, "auto|xdna2|directml|cpu")->capture_default_str();
    f0->add_option("--dump-audio", a->dump_audio, "Write the 16 kHz input as .npy");
    f0->callback([&global, &exit_code, a] { exit_code = run_f0(global, *a); });
}

}  // namespace xr::cli
