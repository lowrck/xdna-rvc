#include "dsp/fft.h"

#include <cmath>
#include <stdexcept>

namespace xr::dsp {

Fft::Fft(int n) : n_(n) {
    if (n < 2 || (n & (n - 1)) != 0) throw std::invalid_argument("Fft size must be a power of two");
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    bitrev_.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
        bitrev_[static_cast<size_t>(i)] = r;
    }
    twiddle_.resize(static_cast<size_t>(n / 2));
    for (int k = 0; k < n / 2; ++k) {
        const double a = -2.0 * 3.14159265358979323846 * k / n;
        twiddle_[static_cast<size_t>(k)] = {static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a))};
    }
}

void Fft::forward(std::span<std::complex<float>> x) const {
    const int n = n_;
    for (int i = 0; i < n; ++i) {
        const int j = bitrev_[static_cast<size_t>(i)];
        if (j > i) std::swap(x[static_cast<size_t>(i)], x[static_cast<size_t>(j)]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len / 2;
        const int step = n / len;
        for (int start = 0; start < n; start += len) {
            for (int k = 0; k < half; ++k) {
                const auto w = twiddle_[static_cast<size_t>(k * step)];
                auto& a = x[static_cast<size_t>(start + k)];
                auto& b = x[static_cast<size_t>(start + k + half)];
                const auto t = w * b;
                b = a - t;
                a = a + t;
            }
        }
    }
}

void Fft::magnitude(std::span<const float> frame, std::span<std::complex<float>> scratch, std::span<float> out) const {
    for (int i = 0; i < n_; ++i) scratch[static_cast<size_t>(i)] = {frame[static_cast<size_t>(i)], 0.0f};
    forward(scratch);
    for (int k = 0; k <= n_ / 2; ++k) out[static_cast<size_t>(k)] = std::abs(scratch[static_cast<size_t>(k)]);
}

}  // namespace xr::dsp
