#pragma once

#include <atomic>

namespace xr::rvc {

// Plain snapshot of the user-adjustable conversion parameters, read once per hop.
struct VoiceParamsSnapshot {
    float pitch_shift = 0.0f;        // semitones
    float index_rate = 0.5f;         // 0 = no retrieval, 1 = retrieved features only
    float rms_mix_rate = 0.25f;      // 1 = keep model loudness, 0 = follow input loudness
    float protect = 0.33f;           // < 0.5 protects unvoiced consonants (F0 models only)
    int speaker_id = 0;
    int filter_radius = 0;           // median filter on F0 (frames, odd >= 3 enables)
    float input_gain_db = 0.0f;
    float output_gain_db = 0.0f;
    float silence_threshold_db = -60.0f;  // input frames below this are gated; <= -60 disables
    bool fill_unvoiced_f0 = true;    // interpolate F0 through unvoiced frames (upstream RVC default)
    bool bypass = false;             // pass the input through instead of converting
};

// Thread-safe parameter block: the GUI/control thread writes individual fields,
// the inference worker takes a snapshot per hop. Lock-free; no field depends on another.
class VoiceParams {
public:
    VoiceParamsSnapshot snapshot() const {
        VoiceParamsSnapshot s;
        s.pitch_shift = pitch_shift.load(std::memory_order_relaxed);
        s.index_rate = index_rate.load(std::memory_order_relaxed);
        s.rms_mix_rate = rms_mix_rate.load(std::memory_order_relaxed);
        s.protect = protect.load(std::memory_order_relaxed);
        s.speaker_id = speaker_id.load(std::memory_order_relaxed);
        s.filter_radius = filter_radius.load(std::memory_order_relaxed);
        s.input_gain_db = input_gain_db.load(std::memory_order_relaxed);
        s.output_gain_db = output_gain_db.load(std::memory_order_relaxed);
        s.silence_threshold_db = silence_threshold_db.load(std::memory_order_relaxed);
        s.fill_unvoiced_f0 = fill_unvoiced_f0.load(std::memory_order_relaxed);
        s.bypass = bypass.load(std::memory_order_relaxed);
        return s;
    }
    void store(const VoiceParamsSnapshot& s) {
        pitch_shift = s.pitch_shift;
        index_rate = s.index_rate;
        rms_mix_rate = s.rms_mix_rate;
        protect = s.protect;
        speaker_id = s.speaker_id;
        filter_radius = s.filter_radius;
        input_gain_db = s.input_gain_db;
        output_gain_db = s.output_gain_db;
        silence_threshold_db = s.silence_threshold_db;
        fill_unvoiced_f0 = s.fill_unvoiced_f0;
        bypass = s.bypass;
    }

    std::atomic<float> pitch_shift{0.0f};
    std::atomic<float> index_rate{0.5f};
    std::atomic<float> rms_mix_rate{0.25f};
    std::atomic<float> protect{0.33f};
    std::atomic<int> speaker_id{0};
    std::atomic<int> filter_radius{0};
    std::atomic<float> input_gain_db{0.0f};
    std::atomic<float> output_gain_db{0.0f};
    std::atomic<float> silence_threshold_db{-60.0f};
    std::atomic<bool> fill_unvoiced_f0{true};
    std::atomic<bool> bypass{false};
};

}  // namespace xr::rvc
