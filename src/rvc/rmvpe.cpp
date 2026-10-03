#include "rvc/rmvpe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "util/error.h"
#include "util/npy.h"

namespace xr::rvc {

namespace {

constexpr int kBinsHalf = RmvpePitch::kNfft / 2 + 1;  // 513
constexpr double kCentsBase = 1997.3794084376191;

}  // namespace

int RmvpePitch::realtime_segment_samples(int block_16k) {
    const int f = block_16k + 800;
    return 5120 * ((f - 1) / 5120 + 1) - 160;
}

RmvpePitch::RmvpePitch(OrtRuntime& runtime, SessionRequest request, OpenOptions options,
                       const std::filesystem::path& mel_basis_npy)
    : fft_(kNfft) {
    if (request.stage.empty()) request.stage = "rmvpe";
    if (options.probe_shapes.empty()) options.probe_shapes["mel"] = {1, kMels, 32};
    const NpyArray basis = read_npy(mel_basis_npy);
    if (basis.shape != std::vector<int64_t>{kMels, kBinsHalf}) {
        throw UserError(mel_basis_npy.string() + " has the wrong shape (expected [128, 513])",
                        "Re-run tools/convert_rmvpe.py.");
    }
    const auto b = basis.as_float();
    mel_basis_.assign(b.begin(), b.end());
    session_ = ModelSession::open(runtime, request, options);
    if (!session_->has_input("mel")) {
        throw UserError("RMVPE model " + request.model_path.string() + " has no input named 'mel'",
                        "Re-create it with tools/convert_rmvpe.py.");
    }
    window_.resize(kNfft);
    for (int i = 0; i < kNfft; ++i) {  // torch.hann_window(1024) is periodic
        window_[static_cast<size_t>(i)] =
            static_cast<float>(0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / kNfft));
    }
    fft_scratch_.resize(kNfft);
    frame_buf_.resize(kNfft);
    mag_buf_.resize(kBinsHalf);
}

void RmvpePitch::compute_log_mel(std::span<const float> audio, int frames, std::span<float> mel) {
    const int n = static_cast<int>(audio.size());
    const int pad = kNfft / 2;
    if (n <= pad) throw std::invalid_argument("RMVPE input shorter than 513 samples");
    padded_.resize(static_cast<size_t>(n + 2 * pad));
    // torch.stft(center=True) reflect padding
    for (int i = 0; i < pad; ++i) padded_[static_cast<size_t>(i)] = audio[static_cast<size_t>(pad - i)];
    std::copy(audio.begin(), audio.end(), padded_.begin() + pad);
    for (int i = 0; i < pad; ++i) padded_[static_cast<size_t>(n + pad + i)] = audio[static_cast<size_t>(n - 2 - i)];

    const int real_frames = frames_for_samples(n);
    for (int f = 0; f < frames; ++f) {
        if (f >= real_frames) {
            // Upstream pads the mel with zeros up to a multiple of 32 frames.
            for (int m = 0; m < kMels; ++m) mel[static_cast<size_t>(m) * frames + f] = 0.0f;
            continue;
        }
        const float* src = padded_.data() + static_cast<size_t>(f) * kHop;
        for (int i = 0; i < kNfft; ++i) frame_buf_[static_cast<size_t>(i)] = src[i] * window_[static_cast<size_t>(i)];
        fft_.magnitude(frame_buf_, fft_scratch_, mag_buf_);
        for (int m = 0; m < kMels; ++m) {
            const float* row = mel_basis_.data() + static_cast<size_t>(m) * kBinsHalf;
            float acc = 0.0f;
            for (int k = 0; k < kBinsHalf; ++k) acc += row[k] * mag_buf_[static_cast<size_t>(k)];
            mel[static_cast<size_t>(m) * frames + f] = std::log(std::max(acc, 1e-5f));
        }
    }
}

void RmvpePitch::decode(std::span<const float> sal, int frames, std::span<float> f0) {
    for (int t = 0; t < frames; ++t) {
        const float* row = sal.data() + static_cast<size_t>(t) * kBins;
        int center = 0;
        float maxv = row[0];
        for (int b = 1; b < kBins; ++b) {
            if (row[b] > maxv) {
                maxv = row[b];
                center = b;
            }
        }
        if (maxv <= kThreshold) {
            f0[static_cast<size_t>(t)] = 0.0f;
            continue;
        }
        // Weighted average of cents over bins center-4 .. center+4 (bins outside 0..359 are 0).
        double num = 0.0, den = 0.0;
        for (int b = center - 4; b <= center + 4; ++b) {
            if (b < 0 || b >= kBins) continue;
            const double cents = 20.0 * b + kCentsBase;
            num += row[b] * cents;
            den += row[b];
        }
        const double cents = num / den;
        const double hz = 10.0 * std::pow(2.0, cents / 1200.0);
        f0[static_cast<size_t>(t)] = static_cast<float>(hz);
    }
}

std::vector<float> RmvpePitch::extract(std::span<const float> audio16k) {
    const int real = frames_for_samples(static_cast<int>(audio16k.size()));
    const int frames = 32 * ((real - 1) / 32 + 1);
    std::vector<float> mel(static_cast<size_t>(kMels) * frames);
    compute_log_mel(audio16k, frames, mel);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 3> shape{1, kMels, frames};
    std::array<Ort::Value, 1> in{Ort::Value::CreateTensor<float>(mem, mel.data(), mel.size(), shape.data(), 3)};
    auto out = session_->run(in);
    const auto oshape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    if (oshape.size() != 3 || oshape[1] != frames || oshape[2] != kBins) {
        throw std::runtime_error("RMVPE produced an unexpected output shape");
    }
    std::vector<float> f0(static_cast<size_t>(real));
    decode(std::span<const float>(out[0].GetTensorData<float>(), static_cast<size_t>(frames) * kBins), real, f0);
    return f0;
}

void RmvpePitch::prepare_fixed(int samples) {
    const int frames = frames_for_samples(samples);
    if (frames % 32 != 0) {
        throw std::invalid_argument("RMVPE fixed segment of " + std::to_string(samples) +
                                    " samples gives " + std::to_string(frames) + " frames, not a multiple of 32");
    }
    fixed_samples_ = samples;
    fixed_frames_ = frames;
    fixed_mel_.assign(static_cast<size_t>(kMels) * frames, 0.0f);
    fixed_hidden_.assign(static_cast<size_t>(frames) * kBins, 0.0f);
    fixed_f0_.assign(static_cast<size_t>(frames), 0.0f);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 3> ishape{1, kMels, frames};
    const std::array<int64_t, 3> oshape{1, frames, kBins};
    bound_in_.clear();
    bound_out_.clear();
    bound_in_.push_back(Ort::Value::CreateTensor<float>(mem, fixed_mel_.data(), fixed_mel_.size(), ishape.data(), 3));
    bound_out_.push_back(
        Ort::Value::CreateTensor<float>(mem, fixed_hidden_.data(), fixed_hidden_.size(), oshape.data(), 3));
    // Size scratch buffers so run_fixed() does not allocate.
    padded_.resize(static_cast<size_t>(samples + kNfft));
}

std::span<const float> RmvpePitch::run_fixed(std::span<const float> audio16k) {
    if (bound_in_.empty()) throw std::logic_error("RmvpePitch::run_fixed() before prepare_fixed()");
    if (static_cast<int>(audio16k.size()) != fixed_samples_) {
        throw std::invalid_argument("RmvpePitch::run_fixed(): wrong segment length");
    }
    compute_log_mel(audio16k, fixed_frames_, fixed_mel_);
    session_->run(bound_in_, bound_out_);
    decode(fixed_hidden_, fixed_frames_, fixed_f0_);
    return fixed_f0_;
}

}  // namespace xr::rvc
