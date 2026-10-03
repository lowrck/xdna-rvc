#include <doctest/doctest.h>

#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <vector>

#include "dsp/fft.h"
#include "dsp/sola.h"
#include "util/latency_stats.h"

using namespace xr;
using namespace xr::dsp;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("FFT magnitude matches a direct DFT") {
    const int n = 64;
    Fft fft(n);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    std::vector<float> x(n), mag(n / 2 + 1);
    for (auto& v : x) v = nd(rng);
    std::vector<std::complex<float>> scratch(n);
    fft.magnitude(x, scratch, mag);
    for (int k = 0; k <= n / 2; ++k) {
        std::complex<double> acc = 0;
        for (int t = 0; t < n; ++t) acc += double(x[t]) * std::polar(1.0, -2 * kPi * k * t / n);
        CHECK(mag[k] == doctest::Approx(std::abs(acc)).epsilon(1e-4));
    }
    CHECK_THROWS_AS(Fft(48), std::invalid_argument);
}

TEST_CASE("SOLA crossfade: a continuous signal stays continuous across blocks") {
    // Feed overlapping windows of one sine as an "inference" would produce them and
    // check the stitched output has no discontinuities (clicks).
    const int block = 400, buffer = 160, search = 40;
    Sola sola(block, buffer, search);
    const double f = 220.0, sr = 16000.0;
    auto sig = [&](long i) { return static_cast<float>(0.5 * std::sin(2 * kPi * f * i / sr)); };
    std::vector<float> in(sola.input_size()), out(block), all;
    for (int b = 0; b < 30; ++b) {
        // Each window starts at the block boundary; jitter the content start by a few
        // samples to emulate phase misalignment the search must correct.
        const long start = static_cast<long>(b) * block + (b % 3) * 7;
        for (int i = 0; i < sola.input_size(); ++i) in[i] = sig(start + i);
        sola.process(in, out);
        all.insert(all.end(), out.begin(), out.end());
    }
    double max_step = 0;
    for (size_t i = block * 2 + 1; i < all.size(); ++i) max_step = std::max(max_step, double(std::abs(all[i] - all[i - 1])));
    const double sine_step = 0.5 * 2 * kPi * f / sr;  // max derivative of the sine per sample
    CHECK(max_step < 1.5 * sine_step);
}

TEST_CASE("SOLA without alignment help still crossfades instead of cutting") {
    Sola sola(100, 40, 0);
    std::vector<float> a(sola.input_size(), 1.0f), b(sola.input_size(), -1.0f), out(100);
    sola.process(a, out);
    sola.process(b, out);
    // First samples ramp from the previous tail (+1) towards the new signal (-1).
    CHECK(out[0] == doctest::Approx(1.0f));
    CHECK(out[39] == doctest::Approx(-1.0f));
    for (int i = 1; i < 40; ++i) CHECK(out[i] <= out[i - 1] + 1e-6f);
}

TEST_CASE("RMS frames follow librosa.feature.rms(center=True) conventions") {
    std::vector<float> x(1600, 0.5f);
    std::vector<float> r;
    RmsMixer::rms_frames(x, 160, r);
    REQUIRE(r.size() == 11);           // 1 + len // hop
    CHECK(r[5] == doctest::Approx(0.5f));
    CHECK(r[0] == doctest::Approx(0.5f * std::sqrt(0.5f)).epsilon(1e-3));  // half the window is padding
}

TEST_CASE("RMS mixing: rate 0 imposes the input envelope, rate 1 leaves output untouched") {
    std::vector<float> in(1600), out(4000);
    for (size_t i = 0; i < in.size(); ++i) in[i] = 0.2f * std::sin(0.05f * i);
    for (size_t i = 0; i < out.size(); ++i) out[i] = 0.8f * std::sin(0.03f * i);
    auto out_keep = out;
    RmsMixer m;
    m.apply(in, 160, out_keep, 400, 1.0f);
    CHECK(out_keep == out);
    m.apply(in, 160, out, 400, 0.0f);
    std::vector<float> r;
    RmsMixer::rms_frames(out, 400, r);
    CHECK(r[r.size() / 2] == doctest::Approx(0.2f / std::sqrt(2.0f)).epsilon(0.05));
}

TEST_CASE("rolling latency statistics") {
    RollingStats s(100);
    for (int i = 1; i <= 100; ++i) s.push(i);
    auto sum = s.summary();
    CHECK(sum.count == 100);
    CHECK(sum.mean == doctest::Approx(50.5));
    CHECK(sum.median == doctest::Approx(50.5));
    CHECK(sum.p95 == doctest::Approx(95));
    CHECK(sum.max == doctest::Approx(100));
    CHECK(sum.last == doctest::Approx(100));
    for (int i = 0; i < 50; ++i) s.push(1000);  // window rolls
    sum = s.summary();
    CHECK(sum.count == 100);
    CHECK(sum.max == doctest::Approx(1000));
    CHECK(sum.median == doctest::Approx((100 + 1000) / 2.0));
    std::vector<float> h;
    s.history(h);
    CHECK(h.size() == 100);
    CHECK(h.back() == doctest::Approx(1000));
    CHECK(h.front() == doctest::Approx(51));
}
