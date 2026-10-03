// GuiApp drawing code (Dear ImGui). No DSP here: the UI writes parameters and reads
// engine snapshots.

#include <algorithm>
#include <cmath>

#include <imgui.h>
#include <spdlog/fmt/fmt.h>

#include "ui/gui_app.h"
#include "util/log.h"

namespace xr::ui {

namespace {

const char* kApiNames[] = {"Default", "WASAPI (shared)", "WASAPI (exclusive)", "PulseAudio", "ALSA", "JACK"};
const char* kBackendNames[] = {"Automatic (XDNA 2 > DirectML > CPU)", "XDNA 2 (NPU)", "DirectML (GPU)", "CPU"};
const char* kStageBackendNames[] = {"Same as Backend", "XDNA 2 (NPU)", "DirectML (GPU)", "CPU"};
const char* kPresetNames[] = {"Low latency (40 ms hop, 60 ms lookahead)", "Balanced (60 ms hop, 60 ms lookahead)",
                              "Quality (100 ms hop, 600 ms context)", "Custom"};
const char* kPriorityNames[] = {"Normal", "High", "Realtime"};
const char* kUnderrunNames[] = {"Silence", "Bypass (dry input)"};

const ImVec4 kGreen{0.35f, 0.85f, 0.45f, 1.0f};
const ImVec4 kAmber{0.95f, 0.75f, 0.25f, 1.0f};
const ImVec4 kRed{0.95f, 0.40f, 0.35f, 1.0f};
const ImVec4 kDim{0.65f, 0.65f, 0.70f, 1.0f};

bool combo(const char* label, int* sel, const char* const* items, int n) { return ImGui::Combo(label, sel, items, n); }

void device_combo(const char* label, int* sel, const std::vector<audio::AudioDeviceInfo>& devs) {
    const char* preview = (*sel >= 0 && *sel < static_cast<int>(devs.size())) ? devs[static_cast<size_t>(*sel)].name.c_str()
                                                                              : "(none)";
    if (ImGui::BeginCombo(label, preview)) {
        for (size_t i = 0; i < devs.size(); ++i) {
            const std::string name = devs[i].name + (devs[i].is_default ? "  (default)" : "");
            if (ImGui::Selectable(name.c_str(), *sel == static_cast<int>(i))) *sel = static_cast<int>(i);
        }
        ImGui::EndCombo();
    }
}

void stat_row(const char* name, const StatsSummary& s) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(name);
    for (double v : {s.mean, s.median, s.p95, s.max}) {
        ImGui::TableNextColumn();
        ImGui::Text("%.1f", v);
    }
}

std::string stage_label(const std::string& s) {
    if (s == "content_encoder") return "Content encoder";
    if (s == "rmvpe") return "Pitch (RMVPE)";
    if (s == "generator") return "Generator";
    return s;
}

}  // namespace

void GuiApp::frame() {
    poll_start();
    if (engine_) engine_->params().store(params_);

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("xdna-rvc", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);
    if (ImGui::BeginTabBar("tabs")) {
        auto tab = [&](const char* name) {
            ImGuiTabItemFlags f = (forced_tab_ == name) ? ImGuiTabItemFlags_SetSelected : 0;
            return ImGui::BeginTabItem(name, nullptr, f);
        };
        if (tab("Main")) {
            draw_main();
            ImGui::EndTabItem();
        }
        if (tab("Advanced")) {
            draw_advanced();
            ImGui::EndTabItem();
        }
        if (tab("Diagnostics")) {
            draw_diagnostics();
            ImGui::EndTabItem();
        }
        if (tab("Import model")) {
            draw_import();
            ImGui::EndTabItem();
        }
        if (tab("Log")) {
            draw_log();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    forced_tab_.clear();
    ImGui::End();
}

void GuiApp::draw_main() {
    const float left_w = ImGui::GetContentRegionAvail().x * 0.48f;
    ImGui::BeginChild("left", ImVec2(left_w, 0), ImGuiChildFlags_None);

    ImGui::SeparatorText("Audio");
    if (combo("Audio API", &audio_api_, kApiNames, IM_ARRAYSIZE(kApiNames))) refresh_devices();
    if (!audio_error_.empty()) ImGui::TextColored(kRed, "%s", audio_error_.c_str());
    ImGui::BeginDisabled(engine_ != nullptr || starting_);
    device_combo("Input device", &input_sel_, inputs_);
    device_combo("Output device", &output_sel_, outputs_);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::SmallButton("Refresh")) refresh_devices();
    ImGui::TextColored(kDim, "Tip: choose a virtual audio cable as output to feed Discord/OBS.");

    ImGui::SeparatorText("Voice model");
    ImGui::BeginDisabled(engine_ != nullptr || starting_);
    combo("Backend", &backend_all_, kBackendNames, IM_ARRAYSIZE(kBackendNames));
    const char* mprev = (model_sel_ >= 0) ? models_[static_cast<size_t>(model_sel_)].name.c_str() : "(no models - import one)";
    if (ImGui::BeginCombo("RVC model", mprev)) {
        for (size_t i = 0; i < models_.size(); ++i) {
            if (ImGui::Selectable(models_[i].name.c_str(), model_sel_ == static_cast<int>(i))) model_sel_ = static_cast<int>(i);
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::SmallButton("Rescan")) refresh_models();
    if (model_sel_ >= 0) {
        const auto& me = models_[static_cast<size_t>(model_sel_)];
        if (me.info) {
            const auto& m = *me.info;
            ImGui::TextColored(kDim, "RVC %s, %d-dim, %s, %d Hz, %d speaker(s)", m.rvc_version.c_str(), m.feature_dim,
                               m.uses_f0 ? "F0" : "no F0", m.sample_rate, m.n_speakers);
            if (m.index) {
                ImGui::TextColored(kDim, "Index file: %lld vectors (IVF%d)", m.index->ntotal, m.index->nlist);
            } else {
                ImGui::TextColored(kAmber, "Index file: none (re-import with an .index to enable retrieval)");
            }
            if (!m.find_generator(rvc::make_geometry(stream_config()))) {
                ImGui::TextColored(kRed, "No generator exported for %s. Change Advanced > Window or re-import.",
                                   stream_config().to_string().c_str());
            }
        } else {
            ImGui::TextColored(kRed, "Invalid model: %s", me.error.c_str());
        }
    }

    ImGui::Spacing();
    if (starting_) {
        ImGui::BeginDisabled();
        ImGui::Button("Starting... (loading models)", ImVec2(-1, 36));
        ImGui::EndDisabled();
    } else if (engine_) {
        if (ImGui::Button("Stop", ImVec2(-1, 36))) stop();
    } else {
        ImGui::BeginDisabled(selected_model() == nullptr);
        if (ImGui::Button("Start", ImVec2(-1, 36))) start();
        ImGui::EndDisabled();
    }
    {
        std::lock_guard lock(start_mu_);
        if (!start_error_.empty()) ImGui::TextColored(kRed, "%s", start_error_.c_str());
    }

    draw_voice_controls();
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("right", ImVec2(0, 0), ImGuiChildFlags_None);
    draw_stats();
    ImGui::EndChild();
}

void GuiApp::draw_voice_controls() {
    ImGui::SeparatorText("Voice");
    ImGui::SliderFloat("Pitch shift (semitones)", &params_.pitch_shift, -24.0f, 24.0f, "%+.1f");
    ImGui::SliderFloat("Index rate", &params_.index_rate, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("RMS mix", &params_.rms_mix_rate, 0.0f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("1 = keep the model's loudness, 0 = follow the microphone's loudness envelope");
    ImGui::SliderFloat("Protect", &params_.protect, 0.0f, 0.5f, "%.2f");
    ImGui::SetItemTooltip("Protects breathy/unvoiced consonants (F0 models). 0.5 disables.");
    ImGui::SliderInt("Filter radius", &params_.filter_radius, 0, 7);
    ImGui::SetItemTooltip("Median filter on the pitch track; 3 or more enables");
    const rvc::ModelInfo* m = selected_model();
    if (m && m->n_speakers > 1) ImGui::SliderInt("Speaker id", &params_.speaker_id, 0, m->n_speakers - 1);
    ImGui::SliderFloat("Input gain (dB)", &params_.input_gain_db, -24.0f, 24.0f, "%+.1f");
    ImGui::SliderFloat("Output gain (dB)", &params_.output_gain_db, -24.0f, 24.0f, "%+.1f");
    ImGui::SliderFloat("Noise gate (dB)", &params_.silence_threshold_db, -60.0f, -20.0f, "%.0f");
    ImGui::SetItemTooltip("Input below this level is muted. -60 disables.");
    ImGui::Checkbox("Bypass (pass input through)", &params_.bypass);
}

void GuiApp::draw_stats() {
    ImGui::SeparatorText("Realtime");
    if (!engine_) {
        ImGui::TextColored(kDim, starting_ ? "Loading models..." : "Stopped.");
        return;
    }
    const auto s = engine_->snapshot();
    bool npu = false;
    if (processor_) {
        for (const auto* r : processor_->stage_reports()) npu = npu || r->npu_evidence();
    }
    if (npu) {
        ImGui::TextColored(kGreen, "XDNA 2 ACTIVE");
        ImGui::SetItemTooltip("At least one stage has confirmed NPU execution (VitisAI report or ORT profile).");
    } else {
        ImGui::TextColored(kDim, "NPU active: no");
    }
    ImGui::TextColored(kDim, "%s", s.device_description.c_str());

    if (ImGui::BeginTable("stages", 3, ImGuiTableFlags_SizingStretchProp)) {
        if (processor_) {
            for (const auto* r : processor_->stage_reports()) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(stage_label(r->stage).c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(r->fell_back ? kAmber : kGreen, "%s", std::string(display_name(r->effective)).c_str());
                ImGui::TableNextColumn();
                if (r->fell_back) {
                    ImGui::TextColored(kAmber, "fallback from %s", std::string(display_name(r->requested)).c_str());
                    ImGui::SetItemTooltip("%s", r->fallback_reason.c_str());
                } else if (r->npu_evidence()) {
                    ImGui::TextColored(kGreen, "NPU nodes confirmed");
                }
            }
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Index");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(processor_ && processor_->has_index() ? "CPU" : "-");
        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Text("Inference latency: median %.1f ms, p95 %.1f ms (hop %d ms)", s.total.median, s.total.p95, s.hop_ms);
    if (s.total.count > 20 && s.total.p95 > 0.9 * s.hop_ms) {
        ImGui::TextColored(kRed, "Processing is too slow for this hop: choose a larger hop (Advanced) or a faster backend.");
    }
    const ImVec4 lat_col = s.latency_ms.median < 100 ? kGreen : (s.latency_ms.median < 200 ? kAmber : kRed);
    ImGui::TextColored(lat_col, "Total latency (measured): median %.0f ms, p95 %.0f ms", s.latency_ms.median,
                       s.latency_ms.p95);
    ImGui::Text("Input buffer %.1f ms | Output buffer %.1f ms | safety %d ms", s.input_ring_ms, s.output_ring_ms,
                s.safety_ms);
    ImGui::TextColored(s.underrun_events ? kAmber : kDim, "Underruns %llu | Overruns %llu samples | Late %llu | Errors %llu",
                       static_cast<unsigned long long>(s.underrun_events),
                       static_cast<unsigned long long>(s.overrun_samples),
                       static_cast<unsigned long long>(s.worker_late_events),
                       static_cast<unsigned long long>(s.inference_errors));
    if (!s.last_error.empty()) ImGui::TextColored(kRed, "Last error: %s", s.last_error.c_str());

    if (!s.total_history.empty()) {
        const float budget = static_cast<float>(s.hop_ms);
        ImGui::PlotLines("##proc", s.total_history.data(), static_cast<int>(s.total_history.size()), 0,
                         fmt::format("processing ms (budget {:.0f})", budget).c_str(), 0.0f, budget * 1.5f,
                         ImVec2(-1, 80));
        ImGui::PlotLines("##lat", s.latency_history.data(), static_cast<int>(s.latency_history.size()), 0,
                         "measured latency ms", 0.0f,
                         std::max(200.0f, *std::max_element(s.latency_history.begin(), s.latency_history.end()) * 1.2f),
                         ImVec2(-1, 80));
    }

    if (ImGui::BeginTable("timing", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Stage (ms)");
        ImGui::TableSetupColumn("mean");
        ImGui::TableSetupColumn("median");
        ImGui::TableSetupColumn("p95");
        ImGui::TableSetupColumn("max");
        ImGui::TableHeadersRow();
        stat_row("Resample in", s.resample_in);
        stat_row("Content encoder", s.content);
        stat_row("Pitch (RMVPE)", s.pitch);
        stat_row("Index lookup", s.index);
        stat_row("Generator", s.generator);
        stat_row("Postprocess", s.post);
        stat_row("Total", s.total);
        ImGui::EndTable();
    }
}

void GuiApp::draw_advanced() {
    ImGui::TextColored(kDim, "Changes apply the next time you press Start.");
    ImGui::BeginDisabled(engine_ != nullptr || starting_);
    ImGui::SeparatorText("Latency / window");
    combo("Stream preset", &preset_, kPresetNames, IM_ARRAYSIZE(kPresetNames));
    if (preset_ == 3) {
        ImGui::InputInt("Hop / block (ms)", &block_ms_, 10, 20);
        ImGui::InputInt("Crossfade (ms)", &crossfade_ms_, 10, 10);
        ImGui::InputInt("Context / extra (ms)", &extra_ms_, 50, 100);
        ImGui::InputInt("Lookahead (ms)", &lookahead_ms_, 10, 20);
        ImGui::TextColored(kDim, "Custom windows need a matching generator export:\n"
                                 "python tools/convert_rvc.py <voice.pth> --stream %d,%d,%d,%d",
                           block_ms_, crossfade_ms_, extra_ms_, lookahead_ms_);
    }
    const auto g = rvc::make_geometry(stream_config());
    ImGui::TextColored(kDim, "Window %d ms, decode %d ms per hop, right context %d ms", g.frames * 10,
                       g.return_length * 10, g.lookahead_frames() * 10);
    ImGui::InputInt("Device buffer period (ms)", &period_ms_, 1, 5);
    ImGui::InputInt("Device buffer periods", &periods_, 1, 1);
    ImGui::InputInt("Output safety buffer (ms, -1 = auto)", &safety_ms_, 5, 10);
    combo("On underrun play", &underrun_policy_, kUnderrunNames, IM_ARRAYSIZE(kUnderrunNames));
    combo("Inference thread priority", &priority_, kPriorityNames, IM_ARRAYSIZE(kPriorityNames));

    ImGui::SeparatorText("Backend override per model stage");
    combo("Content encoder", &backend_content_, kStageBackendNames, IM_ARRAYSIZE(kStageBackendNames));
    combo("Pitch (RMVPE)", &backend_pitch_, kStageBackendNames, IM_ARRAYSIZE(kStageBackendNames));
    combo("Generator", &backend_generator_, kStageBackendNames, IM_ARRAYSIZE(kStageBackendNames));
    ImGui::Checkbox("Allow CPU fallback (always reported)", &allow_fallback_);
    ImGui::SeparatorText("Diagnostics");
    ImGui::Checkbox("ONNX Runtime profiling (session-long, written to diagnostics/)", &keep_profiling_);
    ImGui::EndDisabled();
    if (ImGui::Checkbox("Verbose diagnostics (debug log level)", &verbose_)) {
        spdlog::set_level(verbose_ ? spdlog::level::debug : spdlog::level::info);
    }
}

void GuiApp::draw_diagnostics() {
    auto section = [](const char* t) { ImGui::SeparatorText(t); };
    auto kv = [](const char* k, const std::string& v, const ImVec4* col = nullptr) {
        ImGui::Text("%-18s", k);
        ImGui::SameLine();
        if (col) ImGui::TextColored(*col, "%s", v.c_str());
        else ImGui::TextUnformatted(v.c_str());
    };
    const auto& npu = runtime_->npu();
    section("System");
    kv("CPU:", sys_.cpu_brand);
    kv("OS:", sys_.os);
    kv("NPU:", std::string(to_string(npu.kind)) + (npu.driver_version.empty() ? "" : "  driver " + npu.driver_version),
       npu.present() ? &kGreen : &kDim);
    kv("NPU detail:", npu.detail);
    kv("ORT version:", runtime_->version() + " (" + runtime_->build_flavor() + " package)");
    const bool vitis = runtime_->has_provider("VitisAIExecutionProvider");
    kv("Ryzen AI EP:", vitis ? "Available" : "Not in this build", vitis ? &kGreen : &kAmber);
    kv("DirectML EP:", runtime_->has_provider("DmlExecutionProvider") ? "Available" : "Not in this build");

    section("Model");
    if (const auto* m = selected_model()) {
        kv("Name:", m->name);
        kv("RVC version:", m->rvc_version);
        kv("Feature dim:", std::to_string(m->feature_dim));
        kv("F0:", m->uses_f0 ? "Yes" : "No");
        kv("Output rate:", std::to_string(m->sample_rate) + " Hz");
        kv("XDNA entries:", std::to_string(m->xdna.size()) + " precompiled/prepared");
    } else {
        ImGui::TextColored(kDim, "No model selected.");
    }

    section("Execution");
    if (processor_) {
        for (const auto* r : processor_->stage_reports()) {
            std::string v = std::string(display_name(r->effective));
            if (r->npu_evidence()) v += "  [NPU confirmed]";
            if (r->fell_back) v += "  [FALLBACK: " + r->fallback_reason + "]";
            kv((stage_label(r->stage) + ":").c_str(), v, r->fell_back ? &kAmber : &kGreen);
            for (const auto& u : r->provider_usage) {
                ImGui::TextColored(kDim, "      %s: %d nodes, %.0f%% of node time", u.provider.c_str(), u.node_count,
                                   r->time_fraction(u.provider).value_or(0.0) * 100.0);
            }
            if (r->vitisai_report) {
                ImGui::TextColored(kDim, "      VitisAI report: NPU %d / CPU %d of %d nodes",
                                   r->vitisai_report->nodes_on("NPU"), r->vitisai_report->nodes_on("CPU"),
                                   r->vitisai_report->nodes_total());
            }
        }
        kv("Index:", processor_->has_index() ? "CPU" : "none");
    } else {
        ImGui::TextColored(kDim, "Start the voice changer to see where each stage runs.");
    }

    section("Timing (median)");
    if (engine_) {
        const auto s = engine_->snapshot();
        kv("Capture buffer:", fmt::format("{:.1f} ms", s.input_device_ms));
        kv("Content encoder:", fmt::format("{:.1f} ms", s.content.median));
        kv("Pitch:", fmt::format("{:.1f} ms", s.pitch.median));
        kv("Index:", fmt::format("{:.2f} ms", s.index.median));
        kv("Generator:", fmt::format("{:.1f} ms", s.generator.median));
        kv("Postprocess:", fmt::format("{:.1f} ms", s.post.median));
        kv("Output buffer:", fmt::format("{:.1f} ms ring + {:.1f} ms device", s.output_ring_ms, s.output_device_ms));
        kv("Algorithmic:", fmt::format("{:.1f} ms (crossfade/lookahead + resamplers)", s.algorithmic_ms));
        kv("Measured latency:", fmt::format("median {:.0f} ms, p95 {:.0f} ms", s.latency_ms.median, s.latency_ms.p95));
        kv("Worker priority:", s.worker_priority);
        kv("Drift corrections:", std::to_string(s.drift_corrections));
    } else {
        ImGui::TextColored(kDim, "Not running.");
    }
}

void GuiApp::draw_import() {
    ImGui::TextWrapped("Converts an RVC voice (.pth + optional .index) into ONNX models for this application, "
                       "validating every export against PyTorch. Requires the Python tools (see README). "
                       "Checkpoints are loaded in safe (weights-only) mode.");
    auto input = [](const char* label, std::string& s) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s", s.c_str());
        if (ImGui::InputText(label, buf, sizeof(buf))) s = buf;
    };
    ImGui::BeginDisabled(importing_);
    input("Voice .pth", import_pth_);
    input("Index (.index, optional)", import_index_);
    input("Name (optional)", import_name_);
    ImGui::SeparatorText("Shared models (only needed once)");
    input("HuBERT dir or .pt", import_hubert_);
    input("rmvpe.pt", import_rmvpe_);
    ImGui::SeparatorText("Environment");
    input("Python", python_);
    ImGui::TextColored(kDim, "Project root: %s", opts_.project_root.string().c_str());
    if (ImGui::Button(importing_ ? "Importing..." : "Import", ImVec2(200, 32))) run_import();
    ImGui::EndDisabled();
    if (import_result_ == 1) {
        refresh_models();
        import_result_ = 0;
    }
    std::string out;
    {
        std::lock_guard lock(import_mu_);
        out = import_output_;
    }
    ImGui::BeginChild("import_out", ImVec2(0, 0), ImGuiChildFlags_Borders);
    ImGui::TextUnformatted(out.c_str());
    if (importing_) ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
}

void GuiApp::draw_log() {
    if (ImGui::SmallButton("Clear")) log_->clear();
    ImGui::BeginChild("log", ImVec2(0, 0), ImGuiChildFlags_Borders);
    for (const auto& l : log_->lines()) {
        const ImVec4 c = l.level >= spdlog::level::err ? kRed : (l.level == spdlog::level::warn ? kAmber : kDim);
        ImGui::TextColored(c, "%s", l.text.c_str());
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
}

}  // namespace xr::ui
