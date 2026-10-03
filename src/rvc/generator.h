#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "inference/model_session.h"
#include "rvc/model_info.h"

namespace xr::rvc {

// RVC synthesizer at a fixed streaming geometry (generator_T*_s*_r*.onnx).
// Inputs/outputs are bound once to owned buffers; run() performs no allocation.
class Generator {
public:
    Generator(OrtRuntime& runtime, SessionRequest request, OpenOptions options, const ModelInfo& model,
              const GeneratorVariant& variant);

    int frames() const { return frames_; }
    int flow_frames() const { return flow_frames_; }
    int output_samples() const { return output_samples_; }
    bool uses_f0() const { return uses_f0_; }

    // Writable input buffers.
    std::span<float> phone() { return phone_; }          // [frames, feature_dim]
    std::span<int64_t> pitch() { return pitch_; }        // [frames] coarse 1..255
    std::span<float> pitchf() { return pitchf_; }        // [frames] Hz
    std::span<float> rnd() { return rnd_; }              // [inter_channels, flow_frames] N(0,1)
    std::span<float> noise() { return noise_; }          // [output_samples] N(0,1)
    void set_speaker(int64_t sid) { sid_[0] = sid; }

    std::span<const float> run();
    const ModelSession& session() const { return *session_; }

private:
    std::unique_ptr<ModelSession> session_;
    bool uses_f0_ = false;
    int frames_ = 0;
    int flow_frames_ = 0;
    int output_samples_ = 0;
    std::vector<float> phone_, pitchf_, rnd_, noise_, audio_;
    std::vector<int64_t> pitch_, sid_;
    std::vector<Ort::Value> in_;
    std::vector<Ort::Value> out_;
};

}  // namespace xr::rvc
