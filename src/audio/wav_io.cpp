#include "audio/wav_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <miniaudio.h>

#include "util/error.h"

namespace xr {

AudioBuffer read_audio_file(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) throw UserError("audio file not found: " + path.string());
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 0, 0);
    ma_decoder dec;
#if defined(_WIN32)
    ma_result r = ma_decoder_init_file_w(path.wstring().c_str(), &cfg, &dec);
#else
    ma_result r = ma_decoder_init_file(path.string().c_str(), &cfg, &dec);
#endif
    if (r != MA_SUCCESS) {
        throw UserError("cannot decode audio file " + path.string() + ": " + ma_result_description(r),
                        "Supported formats: WAV, FLAC, MP3.");
    }
    const ma_uint32 channels = dec.outputChannels;
    AudioBuffer out;
    out.sample_rate = static_cast<int>(dec.outputSampleRate);
    out.source_channels = static_cast<int>(channels);

    std::vector<float> chunk(4096 * channels);
    for (;;) {
        ma_uint64 frames_read = 0;
        r = ma_decoder_read_pcm_frames(&dec, chunk.data(), 4096, &frames_read);
        for (ma_uint64 i = 0; i < frames_read; ++i) {
            float s = 0.0f;
            for (ma_uint32 c = 0; c < channels; ++c) s += chunk[i * channels + c];
            out.samples.push_back(s / static_cast<float>(channels));
        }
        if (r != MA_SUCCESS || frames_read == 0) break;
    }
    ma_decoder_uninit(&dec);
    if (r != MA_SUCCESS && r != MA_AT_END) {
        throw UserError("error while decoding " + path.string() + ": " + ma_result_description(r));
    }
    return out;
}

void write_wav(const std::filesystem::path& path, const std::vector<float>& mono, int sample_rate,
               WavSampleFormat format) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    const ma_format f = format == WavSampleFormat::Float32 ? ma_format_f32 : ma_format_s16;
    ma_encoder_config cfg = ma_encoder_config_init(ma_encoding_format_wav, f, 1, static_cast<ma_uint32>(sample_rate));
    ma_encoder enc;
#if defined(_WIN32)
    ma_result r = ma_encoder_init_file_w(path.wstring().c_str(), &cfg, &enc);
#else
    ma_result r = ma_encoder_init_file(path.string().c_str(), &cfg, &enc);
#endif
    if (r != MA_SUCCESS) throw UserError("cannot create WAV file " + path.string() + ": " + ma_result_description(r));

    ma_uint64 written = 0;
    if (format == WavSampleFormat::Float32) {
        r = ma_encoder_write_pcm_frames(&enc, mono.data(), mono.size(), &written);
    } else {
        std::vector<int16_t> pcm(mono.size());
        for (size_t i = 0; i < mono.size(); ++i) {
            const float s = std::clamp(mono[i], -1.0f, 1.0f);
            pcm[i] = static_cast<int16_t>(std::lrint(s * 32767.0f));
        }
        r = ma_encoder_write_pcm_frames(&enc, pcm.data(), pcm.size(), &written);
    }
    ma_encoder_uninit(&enc);
    if (r != MA_SUCCESS || written != mono.size()) {
        throw UserError("failed writing WAV file " + path.string() + ": " + ma_result_description(r));
    }
}

}  // namespace xr
