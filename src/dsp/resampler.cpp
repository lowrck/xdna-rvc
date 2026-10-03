#include "dsp/resampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace xr::dsp {

namespace {

constexpr double kPi = 3.14159265358979323846;

double bessel_i0(double x) {
    // Power series; converges quickly for the beta values used here.
    double sum = 1.0, term = 1.0;
    const double q = x * x / 4.0;
    for (int k = 1; k < 64; ++k) {
        term *= q / (static_cast<double>(k) * k);
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return sum;
}

double sinc(double x) {
    if (std::abs(x) < 1e-12) return 1.0;
    return std::sin(kPi * x) / (kPi * x);
}

struct Params {
    int zero_crossings;
    double beta;
    double rolloff;
};

Params params_for(ResamplerQuality q) {
    switch (q) {
        case ResamplerQuality::Fast: return {8, 6.0, 0.88};
        case ResamplerQuality::Balanced: return {16, 8.0, 0.92};
        case ResamplerQuality::High: return {32, 9.5, 0.95};
    }
    return {16, 8.0, 0.92};
}

}  // namespace

Resampler::Resampler(int in_rate, int out_rate, ResamplerQuality quality) : in_rate_(in_rate), out_rate_(out_rate) {
    if (in_rate <= 0 || out_rate <= 0) throw std::invalid_argument("Resampler: sample rates must be positive");
    const int g = std::gcd(in_rate, out_rate);
    up_ = out_rate / g;
    down_ = in_rate / g;
    const Params p = params_for(quality);

    // Cutoff relative to the input Nyquist: below the lower of the two Nyquist rates.
    const double cutoff = std::min(1.0, static_cast<double>(up_) / down_) * p.rolloff;
    // Zero crossings are 1/cutoff input samples apart; keep Z of them on each side.
    const double half_len = p.zero_crossings / cutoff;
    taps_ = 2 * static_cast<int>(std::ceil(half_len));
    if (up_ == down_) taps_ = 2;  // passthrough path below never uses coefficients

    coeffs_.assign(static_cast<size_t>(up_) * taps_, 0.0f);
    const double i0_beta = bessel_i0(p.beta);
    for (int ph = 0; ph < up_; ++ph) {
        const double frac = static_cast<double>(ph) / up_;
        double sum = 0.0;
        std::vector<double> tmp(taps_);
        for (int j = 0; j < taps_; ++j) {
            // tap j multiplies input index floor(t) - taps/2 + 1 + j; distance from t:
            const double d = static_cast<double>(j - taps_ / 2 + 1) - frac;
            const double x = d / half_len;
            const double w = std::abs(x) >= 1.0 ? 0.0 : bessel_i0(p.beta * std::sqrt(1.0 - x * x)) / i0_beta;
            tmp[j] = sinc(d * cutoff) * w;
            sum += tmp[j];
        }
        for (int j = 0; j < taps_; ++j) coeffs_[static_cast<size_t>(ph) * taps_ + j] = static_cast<float>(tmp[j] / sum);
    }
    work_.assign(static_cast<size_t>(taps_) + kChunk + 8, 0.0f);
    reset();
}

void Resampler::reset() {
    std::fill(work_.begin(), work_.end(), 0.0f);
    // Prime with taps/2-1 zeros of history so output 0 is centered on input 0.
    valid_ = static_cast<size_t>(taps_ / 2 - 1);
    pos_ = taps_ / 2 - 1;
    phase_ = 0;
}

size_t Resampler::max_output(size_t in_frames) const {
    return (in_frames * static_cast<size_t>(up_)) / static_cast<size_t>(down_) + 2;
}

size_t Resampler::process(std::span<const float> in, std::span<float> out) {
    if (up_ == down_) {
        const size_t n = std::min(in.size(), out.size());
        std::memcpy(out.data(), in.data(), n * sizeof(float));
        return n;
    }
    size_t written = 0;
    const int half = taps_ / 2;
    size_t offset = 0;
    while (offset < in.size()) {
        const size_t n = std::min(kChunk, in.size() - offset);
        if (valid_ + n > work_.size()) {
            // Only reachable when `out` is smaller than max_output(in.size()): the
            // remaining input is dropped rather than overflowing the work buffer.
            break;
        }
        std::memcpy(work_.data() + valid_, in.data() + offset, n * sizeof(float));
        valid_ += n;
        offset += n;

        while (pos_ + half < static_cast<int64_t>(valid_) && written < out.size()) {
            const float* x = work_.data() + (pos_ - half + 1);
            const float* h = coeffs_.data() + static_cast<size_t>(phase_) * taps_;
            float acc = 0.0f;
            for (int j = 0; j < taps_; ++j) acc += x[j] * h[j];
            out[written++] = acc;
            phase_ += down_;
            pos_ += phase_ / up_;
            phase_ %= up_;
        }
        // Discard input no longer needed by any future output.
        const int64_t keep_from = pos_ - half + 1;
        if (keep_from > 0) {
            const size_t shift = static_cast<size_t>(std::min<int64_t>(keep_from, static_cast<int64_t>(valid_)));
            std::memmove(work_.data(), work_.data() + shift, (valid_ - shift) * sizeof(float));
            valid_ -= shift;
            pos_ -= static_cast<int64_t>(shift);
        }
    }
    return written;
}

std::vector<float> Resampler::resample(std::span<const float> in, int in_rate, int out_rate,
                                       ResamplerQuality quality) {
    Resampler r(in_rate, out_rate, quality);
    const size_t want = static_cast<size_t>(
        std::llround(static_cast<double>(in.size()) * out_rate / static_cast<double>(in_rate)));
    std::vector<float> out(r.max_output(in.size() + static_cast<size_t>(r.lookahead_input()) + 1));
    size_t n = r.process(in, out);
    // Flush with zeros so the tail is produced.
    std::vector<float> zeros(static_cast<size_t>(r.lookahead_input()) + 1, 0.0f);
    n += r.process(zeros, std::span<float>(out).subspan(n));
    out.resize(std::min(n, want));
    out.resize(want, 0.0f);
    return out;
}

}  // namespace xr::dsp
