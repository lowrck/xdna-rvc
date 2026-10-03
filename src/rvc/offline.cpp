#include "rvc/offline.h"

#include <chrono>
#include <cmath>

namespace xr::rvc {

OfflineResult convert_offline(StreamProcessor& processor, const std::vector<float>& input,
                              const VoiceParamsSnapshot& params, const std::function<void(double)>& progress) {
    processor.reset();
    const int hop_in = processor.hop_input_samples();
    const int hop_out = processor.hop_output_samples();
    const int out_rate = processor.options().output_rate;
    const int in_rate = processor.options().input_rate;
    const size_t delay_out =
        static_cast<size_t>(std::lround(processor.algorithmic_latency_seconds() * out_rate));
    const size_t want_out =
        static_cast<size_t>(std::llround(static_cast<double>(input.size()) * out_rate / in_rate));
    // Enough hops to cover the input plus the algorithmic delay.
    const size_t total_out = want_out + delay_out;
    const int hops = static_cast<int>((total_out + hop_out - 1) / hop_out);

    OfflineResult r;
    r.sample_rate = out_rate;
    r.hops = hops;
    r.audio_seconds = static_cast<double>(input.size()) / in_rate;
    std::vector<float> out(static_cast<size_t>(hops) * hop_out);
    std::vector<float> block(static_cast<size_t>(hop_in));
    RollingStats s_rin(hops), s_c(hops), s_p(hops), s_i(hops), s_g(hops), s_post(hops), s_t(hops);

    const auto t0 = std::chrono::steady_clock::now();
    for (int h = 0; h < hops; ++h) {
        const size_t start = static_cast<size_t>(h) * hop_in;
        for (int i = 0; i < hop_in; ++i) {
            const size_t idx = start + static_cast<size_t>(i);
            block[static_cast<size_t>(i)] = idx < input.size() ? input[idx] : 0.0f;
        }
        HopTimings t;
        processor.process(block, std::span<float>(out.data() + static_cast<size_t>(h) * hop_out, static_cast<size_t>(hop_out)),
                          params, &t);
        s_rin.push(t.resample_in);
        s_c.push(t.content);
        s_p.push(t.pitch);
        s_i.push(t.index);
        s_g.push(t.generator);
        s_post.push(t.post);
        s_t.push(t.total);
        if (progress && (h % 16 == 0 || h + 1 == hops)) progress(static_cast<double>(h + 1) / hops);
    }
    r.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    r.audio.assign(out.begin() + static_cast<std::ptrdiff_t>(std::min(delay_out, out.size())), out.end());
    r.audio.resize(want_out, 0.0f);
    r.resample_in = s_rin.summary();
    r.content = s_c.summary();
    r.pitch = s_p.summary();
    r.index = s_i.summary();
    r.generator = s_g.summary();
    r.post = s_post.summary();
    r.total = s_t.summary();
    return r;
}

}  // namespace xr::rvc
