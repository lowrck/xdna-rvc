#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "audio/realtime_engine.h"
#include "audio/simulated_stream.h"
#include "inference/backend_kind.h"
#include "inference/ort_runtime.h"
#include "rvc/model_info.h"
#include "rvc/voice_params.h"
#include "ui/log_sink.h"
#include "util/system_info.h"

namespace xr::ui {

// The desktop application: state + ImGui panels. All DSP and inference live in the
// realtime engine; this class only configures it and displays its snapshots.
class GuiApp {
public:
    struct Options {
        std::filesystem::path project_root;  // contains tools/ and models/
        int threads = 0;
        std::filesystem::path simulate_input;  // run with simulated audio from this file (testing/demo)
    };

    GuiApp(Options options, std::shared_ptr<RingLogSink> log_sink);
    ~GuiApp();

    void frame();  // draws one frame of UI
    bool wants_exit() const { return false; }

    // For screenshots/tests: select a tab by name ("Main", "Advanced", ...).
    void select_tab(const std::string& name) { forced_tab_ = name; }
    void request_start() { start(); }

private:
    struct ModelEntry {
        std::filesystem::path json;
        std::string name;
        std::optional<rvc::ModelInfo> info;
        std::string error;
    };

    void draw_main();
    void draw_voice_controls();
    void draw_stats();
    void draw_advanced();
    void draw_diagnostics();
    void draw_import();
    void draw_log();

    void refresh_models();
    void refresh_devices();
    void start();
    void stop();
    void poll_start();
    void load_settings();
    void save_settings() const;
    const rvc::ModelInfo* selected_model() const;
    rvc::StreamConfig stream_config() const;
    void run_import();

    Options opts_;
    std::shared_ptr<RingLogSink> log_;
    SystemInfo sys_;
    std::unique_ptr<OrtRuntime> runtime_;

    // models
    std::vector<ModelEntry> models_;
    int model_sel_ = -1;

    // audio
    int audio_api_ = 0;
    std::unique_ptr<audio::AudioBackend> audio_backend_;
    std::vector<audio::AudioDeviceInfo> inputs_, outputs_;
    int input_sel_ = -1, output_sel_ = -1;
    std::string audio_error_;
    int period_ms_ = 10;
    int periods_ = 3;
    int safety_ms_ = -1;
    int underrun_policy_ = 0;
    int priority_ = 2;

    // inference
    int backend_all_ = 0;  // index into backend choices
    int backend_content_ = 0, backend_pitch_ = 0, backend_generator_ = 0;  // 0 = same as "Backend"
    bool allow_fallback_ = true;
    bool keep_profiling_ = false;
    bool verbose_ = false;
    int preset_ = 1;  // low_latency, balanced, quality, custom
    int block_ms_ = 100, crossfade_ms_ = 40, extra_ms_ = 600;

    // voice
    rvc::VoiceParamsSnapshot params_;

    // engine lifecycle
    std::unique_ptr<audio::RealtimeEngine> engine_;
    rvc::StreamProcessor* processor_ = nullptr;
    std::thread starter_;
    std::atomic<bool> starting_{false};
    std::mutex start_mu_;
    std::unique_ptr<audio::RealtimeEngine> started_engine_;  // handed over from starter_
    rvc::StreamProcessor* started_processor_ = nullptr;
    std::string start_error_;

    // import
    std::string import_pth_, import_index_, import_name_, import_hubert_, import_rmvpe_;
    std::string python_ = "python";
    std::thread importer_;
    std::atomic<bool> importing_{false};
    std::atomic<int> import_result_{0};
    std::mutex import_mu_;
    std::string import_output_;

    std::string forced_tab_;
    std::thread simulator_;
    std::atomic<bool> sim_stop_{false};
};

}  // namespace xr::ui
