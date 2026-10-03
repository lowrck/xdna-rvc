// AudioBackend implementation on miniaudio. One capture device and one playback
// device (they may be different hardware, e.g. a USB mic and a virtual cable).

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

#include <spdlog/fmt/fmt.h>
#include <miniaudio.h>

#include "audio/audio_backend.h"
#include "util/error.h"
#include "util/log.h"

namespace xr::audio {

const char* to_string(AudioApi api) {
    switch (api) {
        case AudioApi::Default: return "default";
        case AudioApi::WasapiShared: return "wasapi";
        case AudioApi::WasapiExclusive: return "wasapi-exclusive";
        case AudioApi::Asio: return "asio";
        case AudioApi::PulseAudio: return "pulseaudio";
        case AudioApi::Alsa: return "alsa";
        case AudioApi::Jack: return "jack";
        case AudioApi::CoreAudio: return "coreaudio";
    }
    return "default";
}

AudioApi parse_audio_api(const std::string& t) {
    std::string s(t);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "default" || s == "auto") return AudioApi::Default;
    if (s == "wasapi" || s == "wasapi-shared") return AudioApi::WasapiShared;
    if (s == "wasapi-exclusive" || s == "exclusive") return AudioApi::WasapiExclusive;
    if (s == "asio") return AudioApi::Asio;
    if (s == "pulseaudio" || s == "pulse") return AudioApi::PulseAudio;
    if (s == "alsa") return AudioApi::Alsa;
    if (s == "jack") return AudioApi::Jack;
    if (s == "coreaudio") return AudioApi::CoreAudio;
    throw std::invalid_argument("unknown audio API '" + t + "' (default|wasapi|wasapi-exclusive|pulseaudio|alsa|jack)");
}

namespace {

std::vector<ma_backend> backends_for(AudioApi api) {
    switch (api) {
        case AudioApi::WasapiShared:
        case AudioApi::WasapiExclusive: return {ma_backend_wasapi};
        case AudioApi::PulseAudio: return {ma_backend_pulseaudio};
        case AudioApi::Alsa: return {ma_backend_alsa};
        case AudioApi::Jack: return {ma_backend_jack};
        case AudioApi::CoreAudio: return {ma_backend_coreaudio};
        default: return {};
    }
}

class MaStream;

void capture_cb(ma_device* dev, void*, const void* input, ma_uint32 frames);
void playback_cb(ma_device* dev, void* output, const void*, ma_uint32 frames);

class MaStream final : public AudioStream {
public:
    MaStream(ma_context& ctx, const StreamParams& p, AudioCallbacks& cb) : cb_(cb), exclusive_(p.api == AudioApi::WasapiExclusive) {
        const ma_share_mode share = exclusive_ ? ma_share_mode_exclusive : ma_share_mode_shared;
        int rate = p.sample_rate;
        auto init = [&](ma_device& dev, ma_device_type type, const std::optional<DeviceId>& id) {
            ma_device_config c = ma_device_config_init(type);
            c.performanceProfile = ma_performance_profile_low_latency;
            c.periodSizeInMilliseconds = static_cast<ma_uint32>(std::max(1, p.period_ms));
            c.periods = static_cast<ma_uint32>(std::max(2, p.periods));
            c.sampleRate = static_cast<ma_uint32>(rate);
            c.pUserData = this;
            c.noPreSilencedOutputBuffer = MA_TRUE;  // we write every sample
            c.noClip = MA_TRUE;
            if (type == ma_device_type_capture) {
                c.capture.format = ma_format_f32;
                c.capture.channels = 1;  // miniaudio downmixes without allocating in the callback
                c.capture.shareMode = share;
                c.capture.pDeviceID = id ? reinterpret_cast<const ma_device_id*>(id->bytes.data()) : nullptr;
                c.dataCallback = capture_cb;
            } else {
                c.playback.format = ma_format_f32;
                c.playback.channels = 1;  // mono duplicated to every output channel
                c.playback.shareMode = share;
                c.playback.pDeviceID = id ? reinterpret_cast<const ma_device_id*>(id->bytes.data()) : nullptr;
                c.dataCallback = playback_cb;
            }
            c.wasapi.noAutoConvertSRC = exclusive_ ? MA_TRUE : MA_FALSE;
            const ma_result r = ma_device_init(&ctx, &c, &dev);
            if (r != MA_SUCCESS) {
                throw UserError(fmt::format("cannot open {} device: {}", type == ma_device_type_capture ? "input" : "output",
                                            ma_result_description(r)),
                                exclusive_ ? "Exclusive mode needs a device not in use by other apps and a supported "
                                             "format; try shared mode."
                                           : "Check the device name with `xdna-rvc-cli devices`.");
            }
        };
        init(capture_, ma_device_type_capture, p.input);
        capture_ok_ = true;
        if (rate == 0) {
            rate = static_cast<int>(capture_.sampleRate);
            if (rate % 100 != 0) {
                // e.g. 22050 Hz: reopen both at 48 kHz and let miniaudio convert.
                ma_device_uninit(&capture_);
                capture_ok_ = false;
                rate = 48000;
                init(capture_, ma_device_type_capture, p.input);
                capture_ok_ = true;
            }
        }
        init(playback_, ma_device_type_playback, p.output);
        playback_ok_ = true;
        if (static_cast<int>(playback_.sampleRate) % 100 != 0) {
            ma_device_uninit(&playback_);
            playback_ok_ = false;
            rate = 48000;
            init(playback_, ma_device_type_playback, p.output);
            playback_ok_ = true;
        }
        backend_ = ma_get_backend_name(ctx.backend);
    }
    ~MaStream() override {
        if (capture_ok_) ma_device_uninit(&capture_);
        if (playback_ok_) ma_device_uninit(&playback_);
    }

    void start() override {
        // Playback first so the pre-filled output buffer is consumed as soon as capture starts.
        ma_result r = ma_device_start(&playback_);
        if (r == MA_SUCCESS) r = ma_device_start(&capture_);
        if (r != MA_SUCCESS) throw UserError(std::string("cannot start audio devices: ") + ma_result_description(r));
    }
    void stop() override {
        ma_device_stop(&capture_);
        ma_device_stop(&playback_);
    }
    int input_rate() const override { return static_cast<int>(capture_.sampleRate); }
    int output_rate() const override { return static_cast<int>(playback_.sampleRate); }
    double input_buffer_ms() const override {
        return 1000.0 * capture_.capture.internalPeriodSizeInFrames * capture_.capture.internalPeriods /
               std::max<ma_uint32>(1, capture_.capture.internalSampleRate);
    }
    double output_buffer_ms() const override {
        return 1000.0 * playback_.playback.internalPeriodSizeInFrames * playback_.playback.internalPeriods /
               std::max<ma_uint32>(1, playback_.playback.internalSampleRate);
    }
    std::string description() const override {
        return fmt::format("{}{}: in '{}' {} Hz ({:.1f} ms buffer), out '{}' {} Hz ({:.1f} ms buffer)", backend_,
                           exclusive_ ? " exclusive" : "", capture_.capture.name, input_rate(), input_buffer_ms(),
                           playback_.playback.name, output_rate(), output_buffer_ms());
    }

    AudioCallbacks& cb_;

private:
    ma_device capture_{};
    ma_device playback_{};
    bool capture_ok_ = false, playback_ok_ = false;
    bool exclusive_ = false;
    std::string backend_;
};

void capture_cb(ma_device* dev, void*, const void* input, ma_uint32 frames) {
    auto* s = static_cast<MaStream*>(dev->pUserData);
    if (input) s->cb_.on_capture(static_cast<const float*>(input), frames);
}

void playback_cb(ma_device* dev, void* output, const void*, ma_uint32 frames) {
    auto* s = static_cast<MaStream*>(dev->pUserData);
    s->cb_.on_playback(static_cast<float*>(output), frames);
}

class MaBackend final : public AudioBackend {
public:
    explicit MaBackend(AudioApi api) : api_(api) {
        if (api == AudioApi::Asio) {
            throw UserError("ASIO is not available in this build",
                            "Use WASAPI exclusive mode (--audio-api wasapi-exclusive) for low latency on Windows.");
        }
        const auto list = backends_for(api);
        ma_context_config cfg = ma_context_config_init();
        cfg.threadPriority = ma_thread_priority_realtime;
        const ma_result r = ma_context_init(list.empty() ? nullptr : list.data(), static_cast<ma_uint32>(list.size()),
                                            &cfg, &ctx_);
        if (r != MA_SUCCESS) {
            throw UserError(fmt::format("audio API '{}' is not available: {}", to_string(api), ma_result_description(r)));
        }
    }
    ~MaBackend() override { ma_context_uninit(&ctx_); }

    std::vector<AudioDeviceInfo> devices(bool inputs) override {
        ma_device_info *play = nullptr, *cap = nullptr;
        ma_uint32 np = 0, nc = 0;
        if (ma_context_get_devices(&ctx_, &play, &np, &cap, &nc) != MA_SUCCESS) return {};
        std::vector<AudioDeviceInfo> out;
        const ma_device_info* list = inputs ? cap : play;
        const ma_uint32 n = inputs ? nc : np;
        for (ma_uint32 i = 0; i < n; ++i) {
            AudioDeviceInfo d;
            d.name = list[i].name;
            d.is_default = list[i].isDefault;
            d.is_input = inputs;
            static_assert(sizeof(ma_device_id) <= sizeof(DeviceId::bytes), "DeviceId too small for ma_device_id");
            std::memcpy(d.id.bytes.data(), &list[i].id, sizeof(ma_device_id));
            out.push_back(std::move(d));
        }
        return out;
    }

    std::unique_ptr<AudioStream> open(const StreamParams& params, AudioCallbacks& callbacks) override {
        StreamParams p = params;
        p.api = api_;
        return std::make_unique<MaStream>(ctx_, p, callbacks);
    }

    std::string name() const override { return ma_get_backend_name(ctx_.backend); }

private:
    AudioApi api_;
    ma_context ctx_{};
};

}  // namespace

std::unique_ptr<AudioBackend> make_audio_backend(AudioApi api) { return std::make_unique<MaBackend>(api); }

DeviceId find_device(AudioBackend& backend, bool input, const std::string& name) {
    const auto list = backend.devices(input);
    for (const auto& d : list) {
        if (d.name == name) return d.id;
    }
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    };
    for (const auto& d : list) {
        if (lower(d.name).find(lower(name)) != std::string::npos) return d.id;
    }
    std::string names;
    for (const auto& d : list) names += "\n  - " + d.name;
    throw UserError(fmt::format("no {} device matches '{}'. Available:{}", input ? "input" : "output", name, names));
}

}  // namespace xr::audio
