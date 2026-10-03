#pragma once

#include <filesystem>
#include <vector>

namespace xr {

struct AudioBuffer {
    std::vector<float> samples;  // mono
    int sample_rate = 0;
    int source_channels = 0;
    double duration_seconds() const {
        return sample_rate > 0 ? static_cast<double>(samples.size()) / sample_rate : 0.0;
    }
};

// Decodes WAV/FLAC/MP3 (anything miniaudio supports) to mono float32 at the file's own rate.
AudioBuffer read_audio_file(const std::filesystem::path& path);

enum class WavSampleFormat { Float32, Int16 };
// Writes mono audio as WAV. Int16 output is clipped to [-1, 1].
void write_wav(const std::filesystem::path& path, const std::vector<float>& mono, int sample_rate,
               WavSampleFormat format = WavSampleFormat::Float32);

}  // namespace xr
