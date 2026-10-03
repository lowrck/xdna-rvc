#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "dsp/resampler.h"

using namespace xr::dsp;

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<float> sine(double freq, int rate, size_t n, double amp = 0.5) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = static_cast<float>(amp * std::sin(2 * kPi * freq * i / rate));
    return x;
}

double rms(const std::vector<float>& x, size_t from, size_t to) {
    double s = 0;
    for (size_t i = from; i < to; ++i) s += double(x[i]) * x[i];
    return std::sqrt(s / double(to - from));
}

}  // namespace

TEST_CASE("resample() preserves amplitude and timing of an in-band sine") {
    struct Case {
        int in, out;
    };
    for (Case c : {Case{48000, 16000}, Case{44100, 16000}, Case{40000, 48000}, Case{16000, 48000},
                   Case{48000, 40000}, Case{32000, 44100}}) {
        CAPTURE(c.in);
        CAPTURE(c.out);
        const double f = 1000.0;
        const auto x = sine(f, c.in, static_cast<size_t>(c.in));  // 1 s
        const auto y = Resampler::resample(x, c.in, c.out, ResamplerQuality::High);
        CHECK(y.size() == static_cast<size_t>(c.out));
        // Compare against the ideal sine at the output rate, away from the edges.
        double max_err = 0;
        for (size_t k = c.out / 10; k < y.size() - c.out / 10; ++k) {
            const double ideal = 0.5 * std::sin(2 * kPi * f * k / c.out);
            max_err = std::max(max_err, std::abs(y[k] - ideal));
        }
        CHECK(max_err < 2e-3);
    }
}

TEST_CASE("downsampling rejects content above the output Nyquist") {
    const auto x = sine(12000.0, 48000, 48000);  // aliases to 4 kHz at 16 kHz if unfiltered
    const auto y = Resampler::resample(x, 48000, 16000, ResamplerQuality::Balanced);
    const double att_db = 20 * std::log10(rms(y, 2000, 14000) / (0.5 / std::sqrt(2.0)));
    CHECK(att_db < -60.0);
}

TEST_CASE("streaming in odd-sized chunks equals one-shot processing") {
    const auto x = sine(440.0, 44100, 30000);
    Resampler a(44100, 16000), b(44100, 16000);
    std::vector<float> ya(a.max_output(x.size()));
    ya.resize(a.process(x, ya));

    std::vector<float> yb;
    std::vector<float> buf(b.max_output(1000));
    size_t pos = 0;
    const size_t sizes[] = {1, 7, 128, 441, 999, 3};
    for (size_t i = 0; pos < x.size(); ++i) {
        const size_t n = std::min(sizes[i % 6], x.size() - pos);
        const size_t got = b.process(std::span<const float>(x).subspan(pos, n), buf);
        yb.insert(yb.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(got));
        pos += n;
    }
    REQUIRE(ya.size() == yb.size());
    for (size_t i = 0; i < ya.size(); ++i) REQUIRE(ya[i] == doctest::Approx(yb[i]).epsilon(1e-6));
}

TEST_CASE("streaming output count tracks the rate ratio and lookahead") {
    Resampler r(48000, 16000);
    std::vector<float> in(4800, 0.1f), out(r.max_output(in.size()));
    size_t total = 0;
    for (int i = 0; i < 10; ++i) total += r.process(in, out);
    // 48000 input samples -> 16000 outputs minus the lookahead not yet available.
    const long expected = 16000 - (r.lookahead_input() / 3);
    CHECK(std::abs(static_cast<long>(total) - expected) <= 1);
    CHECK(r.latency_seconds() < 0.002);
}

TEST_CASE("equal rates pass through unchanged") {
    Resampler r(48000, 48000);
    CHECK(r.is_passthrough());
    std::vector<float> in{0.1f, 0.2f, 0.3f}, out(8);
    CHECK(r.process(in, out) == 3);
    CHECK(out[2] == doctest::Approx(0.3f));
}
