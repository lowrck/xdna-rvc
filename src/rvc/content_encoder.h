#pragma once

#include <memory>
#include <span>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "inference/model_session.h"

namespace xr::rvc {

// HuBERT/ContentVec content encoder: 16 kHz audio -> [frames, dim] features at 50 Hz.
// The ONNX model is produced by tools/convert_contentvec.py:
//   input  "audio"    float32[1, samples]
//   output "features" float32[1, frames, dim]   (dim 256 for RVC v1, 768 for v2)
class ContentEncoder {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr int kHop = 320;  // samples per output frame

    ContentEncoder(OrtRuntime& runtime, SessionRequest request, OpenOptions options);

    int feature_dim() const { return dim_; }
    // Output frames for an input of `samples` samples (HuBERT conv stack arithmetic).
    static int frames_for_samples(int samples);

    // Any length; allocates. Returns row-major [frames, dim].
    std::vector<float> extract(std::span<const float> audio16k, int& frames);

    // Fixed-length operation for streaming: after prepare_fixed(n), run_fixed() reuses
    // pre-bound tensors and performs no allocation of its own.
    void prepare_fixed(int samples);
    int fixed_samples() const { return fixed_samples_; }
    int fixed_frames() const { return fixed_frames_; }
    std::span<float> fixed_input() { return fixed_in_; }
    // Runs on fixed_input(); returns a view of [fixed_frames, dim] features.
    std::span<const float> run_fixed();

    const ModelSession& session() const { return *session_; }

private:
    std::unique_ptr<ModelSession> session_;
    int dim_ = 0;
    int fixed_samples_ = 0;
    int fixed_frames_ = 0;
    std::vector<float> fixed_in_;
    std::vector<float> fixed_out_;
    std::vector<Ort::Value> bound_in_;
    std::vector<Ort::Value> bound_out_;
};

}  // namespace xr::rvc
