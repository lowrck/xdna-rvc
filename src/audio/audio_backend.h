#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace xr::audio {

// Called on audio threads. Implementations must be realtime-safe: no allocation,
// no blocking locks, no I/O, no logging.
class AudioCallbacks {
public:
    virtual ~AudioCallbacks() = default;
    virtual void on_capture(const float* mono, uint32_t frames) = 0;
    virtual void on_playback(float* mono, uint32_t frames) = 0;
};

enum class AudioApi {
    Default,          // platform default (WASAPI shared on Windows)
    WasapiShared,
    WasapiExclusive,
    Asio,             // reserved: requires the Steinberg ASIO SDK (not built in this version)
    PulseAudio,
    Alsa,
    Jack,
    CoreAudio,
};

const char* to_string(AudioApi api);
AudioApi parse_audio_api(const std::string& text);  // throws std::invalid_argument

struct DeviceId {
    std::array<uint8_t, 256> bytes{};  // opaque backend id
    bool operator==(const DeviceId&) const = default;
};

struct AudioDeviceInfo {
    std::string name;
    DeviceId id;
    bool is_default = false;
    bool is_input = false;
};

struct StreamParams {
    AudioApi api = AudioApi::Default;
    std::optional<DeviceId> input;   // nullopt = system default
    std::optional<DeviceId> output;
    int sample_rate = 0;             // 0 = device native (must be a multiple of 100 Hz, else 48000 is used)
    int period_ms = 10;              // device buffer period
    int periods = 3;
};

class AudioStream {
public:
    virtual ~AudioStream() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual int input_rate() const = 0;
    virtual int output_rate() const = 0;
    // Device-side buffering reported by the driver (period * periods), milliseconds.
    virtual double input_buffer_ms() const = 0;
    virtual double output_buffer_ms() const = 0;
    virtual std::string description() const = 0;
};

class AudioBackend {
public:
    virtual ~AudioBackend() = default;
    virtual std::vector<AudioDeviceInfo> devices(bool inputs) = 0;
    // Opens (but does not start) capture + playback for `callbacks`.
    virtual std::unique_ptr<AudioStream> open(const StreamParams& params, AudioCallbacks& callbacks) = 0;
    virtual std::string name() const = 0;
};

// miniaudio-based backend for the given API (WASAPI on Windows, PulseAudio/ALSA/JACK on Linux).
std::unique_ptr<AudioBackend> make_audio_backend(AudioApi api);

// Finds a device by exact name, else by case-insensitive substring. Throws UserError listing devices.
DeviceId find_device(AudioBackend& backend, bool input, const std::string& name);

}  // namespace xr::audio
