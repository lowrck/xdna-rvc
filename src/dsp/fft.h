#pragma once

#include <complex>
#include <span>
#include <vector>

namespace xr::dsp {

// Iterative radix-2 FFT for a fixed power-of-two size. Construction precomputes
// twiddles and the bit-reversal table; transforms do not allocate.
class Fft {
public:
    explicit Fft(int n);
    int size() const { return n_; }
    // In-place forward complex FFT of size n.
    void forward(std::span<std::complex<float>> data) const;
    // Magnitude spectrum (n/2 + 1 bins) of a real frame of size n. `scratch` must have n elements.
    void magnitude(std::span<const float> frame, std::span<std::complex<float>> scratch, std::span<float> out) const;

private:
    int n_;
    std::vector<int> bitrev_;
    std::vector<std::complex<float>> twiddle_;
};

}  // namespace xr::dsp
