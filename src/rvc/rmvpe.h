#pragma once

#include <complex>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "dsp/fft.h"
#include "inference/model_session.h"

namespace xr::rvc {

// RMVPE pitch estimator: 16 kHz audio -> f0 (Hz, 0 = unvoiced) at 100 frames/s.
//
// Front end (CPU, identical to upstream RVC infer/rmvpe.py MelSpectrogram):
//   reflect-pad 512, Hann(1024, periodic), hop 160, |STFT|, 128 HTK mel bands (30-8000 Hz,
//   librosa filterbank loaded from rmvpe_mel_basis.npy), log(max(x, 1e-5)).
// Network (ONNX, any backend): mel[1,128,frames] -> salience[1,frames,360], frames % 32 == 0.
// Decode (CPU): argmax, 9-bin local weighted average of cents, threshold 0.03.
class RmvpePitch {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr int kHop = 160;
    static constexpr int kNfft = 1024;
    static constexpr int kMels = 128;
    static constexpr int kBins = 360;
    static constexpr float kThreshold = 0.03f;

    RmvpePitch(OrtRuntime& runtime, SessionRequest request, OpenOptions options,
               const std::filesystem::path& mel_basis_npy);

    // Number of mel/f0 frames for `samples` of audio (center=True framing).
    static int frames_for_samples(int samples) { return samples / kHop + 1; }
    // Segment length the realtime path analyses for a block of `block_16k` samples
    // (upstream rtrvc: 5120 * ceil((block + 800) / 5120) - 160, giving a multiple of 32 frames).
    static int realtime_segment_samples(int block_16k);

    // Any length; allocates. Returns f0 in Hz per frame (frames_for_samples(n)).
    std::vector<float> extract(std::span<const float> audio16k);

    // Fixed-length streaming path. `samples` must give a multiple of 32 frames.
    void prepare_fixed(int samples);
    int fixed_frames() const { return fixed_frames_; }
    // Runs on `audio16k` (size == prepared samples); f0 written to the returned view.
    std::span<const float> run_fixed(std::span<const float> audio16k);

    const ModelSession& session() const { return *session_; }

    // Exposed for tests.
    void compute_log_mel(std::span<const float> audio16k, int frames, std::span<float> mel_out);
    static void decode(std::span<const float> salience, int frames, std::span<float> f0_out);

private:
    std::unique_ptr<ModelSession> session_;
    std::vector<float> mel_basis_;  // [128, 513]
    std::vector<float> window_;     // periodic Hann(1024)
    dsp::Fft fft_;
    std::vector<std::complex<float>> fft_scratch_;
    std::vector<float> frame_buf_;
    std::vector<float> mag_buf_;
    std::vector<float> padded_;

    int fixed_samples_ = 0;
    int fixed_frames_ = 0;
    std::vector<float> fixed_mel_;
    std::vector<float> fixed_hidden_;
    std::vector<float> fixed_f0_;
    std::vector<Ort::Value> bound_in_;
    std::vector<Ort::Value> bound_out_;
};

}  // namespace xr::rvc
