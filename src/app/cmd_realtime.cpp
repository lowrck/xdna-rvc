// devices / realtime: live microphone -> voice conversion -> output device.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>

#include <spdlog/fmt/fmt.h>

#include "app/cli_context.h"
#include "app/pipeline_cli.h"
#include "audio/realtime_engine.h"
#include "audio/simulated_stream.h"
#include "audio/wav_io.h"
#include "dsp/resampler.h"
#include "rvc/model_info.h"
#include "util/error.h"

namespace xr::cli {

namespace {

std::atomic<bool> g_interrupted{false};
void on_sigint(int) { g_interrupted = true; }

int run_devices(const GlobalOptions& global, const std::string& api) {
    LogConfig cfg;
    cfg.console_level = parse_log_level(global.log_level);
    cfg.log_to_file = false;
    LoggingSession logging(cfg);
    auto backend = audio::make_audio_backend(audio::parse_audio_api(api));
    std::printf("Audio API: %s\n", backend->name().c_str());
    for (bool input : {true, false}) {
        std::printf("\n%s devices\n", input ? "Input" : "Output");
        for (const auto& d : backend->devices(input)) {
            std::printf("  %s %s\n", d.is_default ? "*" : " ", d.name.c_str());
        }
    }
    std::printf("\n(* = system default). For Discord/OBS, choose a virtual audio cable as the output device;\n"
                "see README 'Routing to Discord, games and OBS'.\n");
    return 0;
}

struct RealtimeArgs {
    PipelineArgs pipe;
    std::string api = "default";
    std::string input_device, output_device;
    int period_ms = 10;
    int periods = 3;
    int safety_ms = -1;
    std::string underrun = "silence";
    std::string priority = "realtime";
    double duration = 0;
    std::string simulate, record;
};

void print_status(const audio::EngineSnapshot& s) {
    std::printf("hops %6llu | proc med %5.1f p95 %5.1f max %5.1f ms (budget %d) | latency med %5.1f p95 %5.1f ms | "
                "underruns %llu overruns %llu late %llu errors %llu | out ring %4.1f ms\n",
                static_cast<unsigned long long>(s.hops), s.total.median, s.total.p95, s.total.max, s.hop_ms,
                s.latency_ms.median, s.latency_ms.p95, static_cast<unsigned long long>(s.underrun_events),
                static_cast<unsigned long long>(s.overrun_samples), static_cast<unsigned long long>(s.worker_late_events),
                static_cast<unsigned long long>(s.inference_errors), s.output_ring_ms);
    std::fflush(stdout);
}

int run_realtime(const GlobalOptions& global, const RealtimeArgs& a) {
    CliContext ctx(global, /*realtime=*/true);
    const auto model = rvc::load_model_info(a.pipe.model);
    XR_LOG_INFO("model: {}", model.describe());

    audio::EngineOptions eo;
    eo.safety_ms = a.safety_ms;
    eo.underrun_policy = a.underrun == "bypass" ? audio::UnderrunPolicy::Bypass : audio::UnderrunPolicy::Silence;
    eo.worker_priority = parse_thread_priority(a.priority);

    rvc::ProcessorOptions po;
    po.stream = a.pipe.stream();
    po.backends = a.pipe.backends();
    po.seed = a.pipe.seed;
    po.resampler_quality = a.pipe.resampler_quality();
    rvc::StreamProcessor* proc_ptr = nullptr;
    auto factory = [&](int in_rate, int out_rate) {
        rvc::ProcessorOptions p = po;
        p.input_rate = in_rate;
        p.output_rate = out_rate;
        auto proc = std::make_unique<rvc::StreamProcessor>(ctx.runtime(), model, p);
        proc_ptr = proc.get();
        return proc;
    };

    std::unique_ptr<audio::AudioBackend> backend;
    std::unique_ptr<audio::RealtimeEngine> engine;
    std::vector<float> sim_input;
    if (!a.simulate.empty()) {
        const AudioBuffer wav = read_audio_file(a.simulate);
        sim_input = dsp::Resampler::resample(wav.samples, wav.sample_rate, 48000);
        engine = std::make_unique<audio::RealtimeEngine>(factory, 48000, 48000, 10.0, 10.0, eo);
    } else {
        backend = audio::make_audio_backend(audio::parse_audio_api(a.api));
        audio::StreamParams sp;
        sp.api = audio::parse_audio_api(a.api);
        sp.period_ms = a.period_ms;
        sp.periods = a.periods;
        if (!a.input_device.empty()) sp.input = audio::find_device(*backend, true, a.input_device);
        if (!a.output_device.empty()) sp.output = audio::find_device(*backend, false, a.output_device);
        engine = std::make_unique<audio::RealtimeEngine>(factory, *backend, sp, eo);
    }
    engine->params().store(a.pipe.params());
    if (proc_ptr) print_stage_reports(*proc_ptr);

    std::signal(SIGINT, on_sigint);
    engine->start();
    const auto snap0 = engine->snapshot();
    std::printf("\nRunning: %s. Ctrl+C to stop.\n\n", snap0.device_description.c_str());

    if (!a.simulate.empty()) {
        const double secs = a.duration > 0 ? a.duration : static_cast<double>(sim_input.size()) / 48000.0 + 1.0;
        audio::SimulatedDuplex sim(*engine, sim_input, {48000, 48000, 10, 0.0, true});
        std::atomic<bool> done{false};
        std::thread t([&] {
            sim.run(secs);
            done = true;
        });
        while (!g_interrupted && !done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            static int ticks = 0;
            if (++ticks % 4 == 0) print_status(engine->snapshot());
        }
        t.join();
        engine->stop();
        if (!a.record.empty()) {
            write_wav(a.record, sim.recorded(), 48000);
            std::printf("recorded %s\n", a.record.c_str());
        }
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        while (!g_interrupted) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            print_status(engine->snapshot());
            if (a.duration > 0 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= a.duration)
                break;
        }
        engine->stop();
    }
    const auto s = engine->snapshot();
    std::printf("\nSummary\n-------\n");
    std::printf("  processing per hop: median %.1f ms, p95 %.1f ms, max %.1f ms (hop %d ms)\n", s.total.median,
                s.total.p95, s.total.max, s.hop_ms);
    std::printf("  measured latency:   median %.1f ms, p95 %.1f ms (devices %.1f + %.1f ms, algorithmic %.1f ms)\n",
                s.latency_ms.median, s.latency_ms.p95, s.input_device_ms, s.output_device_ms, s.algorithmic_ms);
    if (s.total.p95 > 0.9 * s.hop_ms) {
        std::printf("  WARNING: p95 processing time is within 10%% of the hop; expect underruns. Use a larger hop "
                    "(--preset balanced/quality) or a faster backend.\n");
    }
    std::printf("  worker priority: %s\n", s.worker_priority.c_str());
    std::printf("  latency trims during silence %llu\n", static_cast<unsigned long long>(s.latency_trims));
    std::printf("  underruns %llu (%llu samples), input overruns %llu samples, worker late %llu, errors %llu, drift "
                "corrections %llu\n",
                static_cast<unsigned long long>(s.underrun_events), static_cast<unsigned long long>(s.underrun_samples),
                static_cast<unsigned long long>(s.overrun_samples), static_cast<unsigned long long>(s.worker_late_events),
                static_cast<unsigned long long>(s.inference_errors), static_cast<unsigned long long>(s.drift_corrections));
    return s.inference_errors > 0 ? 1 : 0;
}

}  // namespace

void register_realtime(CLI::App& app, GlobalOptions& global, int& exit_code) {
    auto api = std::make_shared<std::string>("default");
    auto* dev = app.add_subcommand("devices", "List audio input/output devices");
    dev->add_option("--audio-api", *api, "default|wasapi|wasapi-exclusive|pulseaudio|alsa|jack")->capture_default_str();
    dev->callback([&global, &exit_code, api] { exit_code = run_devices(global, *api); });

    auto a = std::make_shared<RealtimeArgs>();
    auto* rt = app.add_subcommand("realtime", "Live voice conversion from an input device to an output device");
    a->pipe.add_to(*rt, true);
    rt->add_option("--audio-api", a->api, "default|wasapi|wasapi-exclusive|pulseaudio|alsa|jack")->capture_default_str();
    rt->add_option("--input-device", a->input_device, "Input device name (substring match); default device if empty");
    rt->add_option("--output-device", a->output_device, "Output device name, e.g. a virtual cable");
    rt->add_option("--period-ms", a->period_ms, "Device buffer period")->capture_default_str();
    rt->add_option("--periods", a->periods, "Device buffer periods")->capture_default_str();
    rt->add_option("--safety-ms", a->safety_ms, "Output safety buffer beyond one hop (-1 = automatic from measured processing time)")->capture_default_str();
    rt->add_option("--underrun", a->underrun, "On underrun play: silence|bypass")->capture_default_str();
    rt->add_option("--priority", a->priority, "Worker thread priority: normal|high|realtime")->capture_default_str();
    rt->add_option("--duration", a->duration, "Stop after N seconds (0 = until Ctrl+C)");
    rt->add_option("--simulate", a->simulate, "Use this audio file as a realtime-paced input instead of a device");
    rt->add_option("--record", a->record, "With --simulate: write the realtime output to this WAV");
    rt->callback([&global, &exit_code, a] { exit_code = run_realtime(global, *a); });
}

}  // namespace xr::cli
