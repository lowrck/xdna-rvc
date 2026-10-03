// GuiApp: lifecycle, configuration and background jobs. Drawing is in gui_panels.cpp.

#include "ui/gui_app.h"

#include <algorithm>
#include <fstream>

#include <nlohmann/json.hpp>

#include "audio/wav_io.h"
#include "dsp/resampler.h"
#include "ui/process.h"
#include "util/error.h"
#include "util/log.h"

namespace xr::ui {

namespace {

constexpr const char* kSettingsFile = "xdna-rvc-settings.json";
const audio::AudioApi kApis[] = {audio::AudioApi::Default, audio::AudioApi::WasapiShared,
                                 audio::AudioApi::WasapiExclusive, audio::AudioApi::PulseAudio,
                                 audio::AudioApi::Alsa, audio::AudioApi::Jack};
const BackendKind kBackends[] = {BackendKind::Auto, BackendKind::XDNA2, BackendKind::DirectML, BackendKind::CPU};

}  // namespace

GuiApp::GuiApp(Options options, std::shared_ptr<RingLogSink> log_sink) : opts_(std::move(options)), log_(std::move(log_sink)) {
    sys_ = query_system_info();
    int threads = opts_.threads > 0 ? opts_.threads : std::clamp(static_cast<int>(sys_.logical_cores) / 4, 1, 8);
    runtime_ = std::make_unique<OrtRuntime>(ORT_LOGGING_LEVEL_WARNING, threads);
#if defined(_WIN32)
    const auto venv_python = opts_.project_root / ".venv" / "Scripts" / "python.exe";
#else
    const auto venv_python = opts_.project_root / ".venv" / "bin" / "python";
#endif
    if (std::filesystem::exists(venv_python)) python_ = venv_python.string();
    load_settings();
    refresh_models();
    refresh_devices();
}

GuiApp::~GuiApp() {
    save_settings();
    if (starter_.joinable()) starter_.join();
    if (importer_.joinable()) importer_.join();
    stop();
}

void GuiApp::refresh_models() {
    const std::string previous = (model_sel_ >= 0 && model_sel_ < static_cast<int>(models_.size()))
                                     ? models_[static_cast<size_t>(model_sel_)].name
                                     : std::string();
    models_.clear();
    const auto dir = opts_.project_root / "models";
    std::error_code ec;
    if (std::filesystem::is_directory(dir, ec)) {
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            const auto mj = e.path() / "model.json";
            if (!std::filesystem::exists(mj)) continue;
            ModelEntry m;
            m.json = mj;
            m.name = e.path().filename().string();
            try {
                m.info = rvc::load_model_info(mj);
            } catch (const std::exception& ex) {
                m.error = ex.what();
            }
            models_.push_back(std::move(m));
        }
    }
    std::sort(models_.begin(), models_.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    model_sel_ = models_.empty() ? -1 : 0;
    for (size_t i = 0; i < models_.size(); ++i) {
        if (models_[i].name == previous) model_sel_ = static_cast<int>(i);
    }
}

void GuiApp::refresh_devices() {
    inputs_.clear();
    outputs_.clear();
    audio_error_.clear();
    try {
        audio_backend_ = audio::make_audio_backend(kApis[audio_api_]);
        inputs_ = audio_backend_->devices(true);
        outputs_ = audio_backend_->devices(false);
    } catch (const std::exception& e) {
        audio_error_ = e.what();
        audio_backend_.reset();
    }
    auto pick_default = [](const std::vector<audio::AudioDeviceInfo>& v, int cur) {
        if (cur >= 0 && cur < static_cast<int>(v.size())) return cur;
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i].is_default) return static_cast<int>(i);
        }
        return v.empty() ? -1 : 0;
    };
    input_sel_ = pick_default(inputs_, input_sel_);
    output_sel_ = pick_default(outputs_, output_sel_);
}

const rvc::ModelInfo* GuiApp::selected_model() const {
    if (model_sel_ < 0 || model_sel_ >= static_cast<int>(models_.size())) return nullptr;
    const auto& m = models_[static_cast<size_t>(model_sel_)];
    return m.info ? &*m.info : nullptr;
}

rvc::StreamConfig GuiApp::stream_config() const {
    switch (preset_) {
        case 0: return rvc::preset("low_latency");
        case 1: return rvc::preset("balanced");
        case 2: return rvc::preset("quality");
        default: return {block_ms_, crossfade_ms_, extra_ms_};
    }
}

void GuiApp::start() {
    const rvc::ModelInfo* model = selected_model();
    if (!model || starting_ || engine_) return;
    if (!audio_backend_ && opts_.simulate_input.empty()) {
        start_error_ = "no audio API available: " + audio_error_;
        return;
    }
    start_error_.clear();
    starting_ = true;
    const BackendKind all = kBackends[backend_all_];
    auto pick = [&](int sel) { return sel == 0 ? all : kBackends[sel]; };
    rvc::ProcessorOptions po;
    po.stream = stream_config();
    po.backends.content = pick(backend_content_);
    po.backends.pitch = pick(backend_pitch_);
    po.backends.generator = pick(backend_generator_);
    po.backends.allow_cpu_fallback = allow_fallback_;
    po.backends.keep_profiling = keep_profiling_;
    po.backends.cache_dir = opts_.project_root / "cache";
    po.backends.diagnostics_dir = opts_.project_root / "diagnostics";
    audio::EngineOptions eo;
    eo.safety_ms = safety_ms_;
    eo.underrun_policy = underrun_policy_ == 1 ? audio::UnderrunPolicy::Bypass : audio::UnderrunPolicy::Silence;
    eo.worker_priority = static_cast<ThreadPriority>(priority_);
    audio::StreamParams sp;
    sp.api = kApis[audio_api_];
    sp.period_ms = period_ms_;
    sp.periods = periods_;
    if (input_sel_ >= 0) sp.input = inputs_[static_cast<size_t>(input_sel_)].id;
    if (output_sel_ >= 0) sp.output = outputs_[static_cast<size_t>(output_sel_)].id;
    const rvc::ModelInfo model_copy = *model;
    const auto params = params_;
    if (starter_.joinable()) starter_.join();
    // Loading models (and possibly compiling for the NPU) takes seconds: do it off the UI thread.
    starter_ = std::thread([this, po, eo, sp, model_copy, params] {
        try {
            rvc::StreamProcessor* proc = nullptr;
            auto factory = [&](int in_rate, int out_rate) {
                rvc::ProcessorOptions p = po;
                p.input_rate = in_rate;
                p.output_rate = out_rate;
                auto ptr = std::make_unique<rvc::StreamProcessor>(*runtime_, model_copy, p);
                proc = ptr.get();
                return ptr;
            };
            std::unique_ptr<audio::RealtimeEngine> engine;
            if (opts_.simulate_input.empty()) {
                engine = std::make_unique<audio::RealtimeEngine>(factory, *audio_backend_, sp, eo);
            } else {
                engine = std::make_unique<audio::RealtimeEngine>(factory, 48000, 48000, 10.0, 10.0, eo);
            }
            engine->params().store(params);
            engine->start();
            if (!opts_.simulate_input.empty()) {
                // Feed the file in a loop with realtime pacing, as a microphone would.
                const AudioBuffer wav = read_audio_file(opts_.simulate_input);
                auto input = dsp::Resampler::resample(wav.samples, wav.sample_rate, 48000);
                auto* eng = engine.get();
                sim_stop_ = false;
                simulator_ = std::thread([this, eng, input] {
                    while (!sim_stop_) {
                        audio::SimulatedDuplex sim(*eng, input, {48000, 48000, 10, 0.0, true});
                        sim.run(static_cast<double>(input.size()) / 48000.0);
                    }
                });
            }
            std::lock_guard lock(start_mu_);
            started_engine_ = std::move(engine);
            started_processor_ = proc;
        } catch (const UserError& e) {
            std::lock_guard lock(start_mu_);
            start_error_ = e.what();
            if (!e.hint().empty()) start_error_ += "\nHint: " + e.hint();
            XR_LOG_ERROR("start failed: {}", e.what());
        } catch (const std::exception& e) {
            std::lock_guard lock(start_mu_);
            start_error_ = e.what();
            XR_LOG_ERROR("start failed: {}", e.what());
        }
        starting_ = false;
    });
}

void GuiApp::poll_start() {
    std::lock_guard lock(start_mu_);
    if (started_engine_) {
        engine_ = std::move(started_engine_);
        processor_ = started_processor_;
        started_processor_ = nullptr;
    }
}

void GuiApp::stop() {
    sim_stop_ = true;
    if (simulator_.joinable()) simulator_.join();
    if (engine_) {
        engine_->stop();
        engine_.reset();
        processor_ = nullptr;
    }
}

void GuiApp::run_import() {
    if (importing_ || import_pth_.empty()) return;
    if (importer_.joinable()) importer_.join();
    importing_ = true;
    {
        std::lock_guard lock(import_mu_);
        import_output_.clear();
    }
    importer_ = std::thread([this] {
        auto add = [this](const std::string& line) {
            std::lock_guard lock(import_mu_);
            import_output_ += line + "\n";
        };
        int rc = 0;
        try {
            const auto tools = opts_.project_root / "tools";
            const auto shared = opts_.project_root / "models" / "shared";
            if (!import_hubert_.empty()) {
                add("== converting HuBERT/ContentVec content encoders");
                rc = run_process({python_, (tools / "convert_contentvec.py").string(), "--hubert", import_hubert_,
                                  "--out-dir", shared.string(), "--version", "v1", "--version", "v2"}, add);
            }
            if (rc == 0 && !import_rmvpe_.empty()) {
                add("== converting RMVPE");
                rc = run_process({python_, (tools / "convert_rmvpe.py").string(), "--rmvpe", import_rmvpe_, "--out-dir",
                                  shared.string()}, add);
            }
            if (rc == 0) {
                add("== converting voice model");
                std::vector<std::string> args{python_, (tools / "convert_rvc.py").string(), import_pth_, "--models-dir",
                                              (opts_.project_root / "models").string()};
                if (!import_index_.empty()) args.insert(args.end(), {"--index", import_index_});
                if (!import_name_.empty()) args.insert(args.end(), {"--name", import_name_});
                rc = run_process(args, add);
            }
        } catch (const std::exception& e) {
            add(std::string("error: ") + e.what());
            rc = -1;
        }
        add(rc == 0 ? "== import finished" : "== import FAILED (exit code " + std::to_string(rc) + ")");
        import_result_ = rc == 0 ? 1 : -1;
        importing_ = false;
    });
}

void GuiApp::load_settings() {
    std::ifstream in(opts_.project_root / kSettingsFile);
    if (!in) return;
    try {
        nlohmann::json j;
        in >> j;
        audio_api_ = std::clamp(j.value("audio_api", 0), 0, 5);
        backend_all_ = std::clamp(j.value("backend", 0), 0, 3);
        backend_content_ = std::clamp(j.value("backend_content", 0), 0, 3);
        backend_pitch_ = std::clamp(j.value("backend_pitch", 0), 0, 3);
        backend_generator_ = std::clamp(j.value("backend_generator", 0), 0, 3);
        preset_ = std::clamp(j.value("preset", 1), 0, 3);
        block_ms_ = j.value("block_ms", 100);
        crossfade_ms_ = j.value("crossfade_ms", 40);
        extra_ms_ = j.value("extra_ms", 600);
        period_ms_ = j.value("period_ms", 10);
        safety_ms_ = j.value("safety_ms", -1);
        params_.pitch_shift = j.value("pitch", 0.0f);
        params_.index_rate = j.value("index_rate", 0.5f);
        params_.rms_mix_rate = j.value("rms_mix", 0.25f);
        params_.protect = j.value("protect", 0.33f);
        params_.input_gain_db = j.value("input_gain_db", 0.0f);
        params_.output_gain_db = j.value("output_gain_db", 0.0f);
        params_.silence_threshold_db = j.value("gate_db", -60.0f);
        python_ = j.value("python", python_);
    } catch (const std::exception& e) {
        XR_LOG_WARN("ignoring unreadable settings file: {}", e.what());
    }
}

void GuiApp::save_settings() const {
    nlohmann::json j{{"audio_api", audio_api_}, {"backend", backend_all_}, {"backend_content", backend_content_},
                     {"backend_pitch", backend_pitch_}, {"backend_generator", backend_generator_},
                     {"preset", preset_}, {"block_ms", block_ms_}, {"crossfade_ms", crossfade_ms_},
                     {"extra_ms", extra_ms_}, {"period_ms", period_ms_}, {"safety_ms", safety_ms_},
                     {"pitch", params_.pitch_shift}, {"index_rate", params_.index_rate},
                     {"rms_mix", params_.rms_mix_rate}, {"protect", params_.protect},
                     {"input_gain_db", params_.input_gain_db}, {"output_gain_db", params_.output_gain_db},
                     {"gate_db", params_.silence_threshold_db}, {"python", python_}};
    std::ofstream(opts_.project_root / kSettingsFile) << j.dump(2);
}

}  // namespace xr::ui
