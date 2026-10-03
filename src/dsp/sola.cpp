#include "dsp/sola.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xr::dsp {

Sola::Sola(int block, int buffer, int search) : block_(block), buffer_(buffer), search_(search) {
    if (block <= 0 || buffer <= 0 || search < 0) throw std::invalid_argument("Sola: invalid sizes");
    tail_.assign(static_cast<size_t>(buffer), 0.0f);
    fade_in_.resize(static_cast<size_t>(buffer));
    for (int i = 0; i < buffer; ++i) {
        // torch.sin(0.5 * pi * linspace(0, 1, buffer)) ** 2
        const double t = buffer > 1 ? static_cast<double>(i) / (buffer - 1) : 1.0;
        const double s = std::sin(0.5 * 3.14159265358979323846 * t);
        fade_in_[static_cast<size_t>(i)] = static_cast<float>(s * s);
    }
}

void Sola::reset() { std::fill(tail_.begin(), tail_.end(), 0.0f); }

int Sola::process(std::span<const float> in, std::span<float> out) {
    // Normalised cross-correlation of the previous tail against candidate offsets.
    int best = 0;
    double best_score = -1e300;
    for (int k = 0; k <= search_; ++k) {
        double nom = 0.0, den = 0.0;
        for (int i = 0; i < buffer_; ++i) {
            const double x = in[static_cast<size_t>(k + i)];
            nom += x * tail_[static_cast<size_t>(i)];
            den += x * x;
        }
        const double score = nom / std::sqrt(den + 1e-8);
        if (score > best_score) {
            best_score = score;
            best = k;
        }
    }
    const float* x = in.data() + best;
    for (int i = 0; i < block_; ++i) {
        float v = x[i];
        if (i < buffer_) {
            const float fi = fade_in_[static_cast<size_t>(i)];
            v = v * fi + tail_[static_cast<size_t>(i)] * (1.0f - fi);
        }
        out[static_cast<size_t>(i)] = v;
    }
    // If the block is shorter than the crossfade the remaining fade continues in the tail.
    for (int i = 0; i < buffer_; ++i) tail_[static_cast<size_t>(i)] = x[block_ + i];
    return best;
}

void RmsMixer::rms_frames(std::span<const float> x, int hop, std::vector<float>& out) {
    const int frame = 4 * hop;
    const int n = static_cast<int>(x.size());
    const int frames = 1 + n / hop;
    out.resize(static_cast<size_t>(frames));
    const int pad = frame / 2;
    for (int f = 0; f < frames; ++f) {
        const int start = f * hop - pad;
        double acc = 0.0;
        for (int i = 0; i < frame; ++i) {
            const int idx = start + i;
            if (idx >= 0 && idx < n) acc += static_cast<double>(x[static_cast<size_t>(idx)]) * x[static_cast<size_t>(idx)];
        }
        out[static_cast<size_t>(f)] = static_cast<float>(std::sqrt(acc / frame));
    }
}

namespace {

// F.interpolate(env[None, None], size=n + 1, mode="linear", align_corners=True)[..., :-1]
float interp_aligned(const std::vector<float>& env, int i, int n) {
    if (env.size() == 1) return env[0];
    const double pos = static_cast<double>(i) * (env.size() - 1) / static_cast<double>(n);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, env.size() - 1);
    const double t = pos - static_cast<double>(lo);
    return static_cast<float>(env[lo] * (1.0 - t) + env[hi] * t);
}

}  // namespace

void RmsMixer::apply(std::span<const float> input, int in_hop, std::span<float> output, int out_hop, float rate) {
    if (rate >= 1.0f || output.empty()) return;
    rms_frames(input, in_hop, rms_in_);
    rms_frames(output, out_hop, rms_out_);
    const int n = static_cast<int>(output.size());
    const float expo = 1.0f - rate;
    for (int i = 0; i < n; ++i) {
        const float a = interp_aligned(rms_in_, i, n);
        const float b = std::max(interp_aligned(rms_out_, i, n), 1e-3f);
        output[static_cast<size_t>(i)] *= std::pow(a / b, expo);
    }
}

}  // namespace xr::dsp
