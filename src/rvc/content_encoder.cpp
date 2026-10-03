#include "rvc/content_encoder.h"

#include <array>
#include <stdexcept>

#include "util/error.h"

namespace xr::rvc {

namespace {
constexpr std::array<int, 7> kKernel = {10, 3, 3, 3, 3, 2, 2};
constexpr std::array<int, 7> kStride = {5, 2, 2, 2, 2, 2, 2};
}  // namespace

int ContentEncoder::frames_for_samples(int samples) {
    int n = samples;
    for (size_t i = 0; i < kKernel.size(); ++i) {
        n = (n - kKernel[i]) / kStride[i] + 1;
        if (n <= 0) return 0;
    }
    return n;
}

ContentEncoder::ContentEncoder(OrtRuntime& runtime, SessionRequest request, OpenOptions options) {
    if (request.stage.empty()) request.stage = "content_encoder";
    if (options.probe_shapes.empty()) options.probe_shapes["audio"] = {1, kSampleRate};
    session_ = ModelSession::open(runtime, request, options);
    if (!session_->has_input("audio") || session_->outputs().empty()) {
        throw UserError("content encoder model " + request.model_path.string() +
                            " does not have the expected input 'audio' / output 'features'",
                        "Re-create it with tools/convert_contentvec.py.");
    }
    const auto& out = session_->output("features");
    if (out.shape.size() != 3 || out.shape[2] <= 0) {
        throw UserError("content encoder output 'features' must be [1, frames, dim] with static dim, got " +
                        out.to_string());
    }
    dim_ = static_cast<int>(out.shape[2]);
}

std::vector<float> ContentEncoder::extract(std::span<const float> audio16k, int& frames) {
    frames = frames_for_samples(static_cast<int>(audio16k.size()));
    if (frames <= 0) throw std::invalid_argument("content encoder input too short (" +
                                                 std::to_string(audio16k.size()) + " samples)");
    const auto& in = session_->input("audio");
    if (in.is_static() && in.shape[1] != static_cast<int64_t>(audio16k.size())) {
        throw std::invalid_argument("content encoder model has fixed input length " + std::to_string(in.shape[1]) +
                                    ", got " + std::to_string(audio16k.size()));
    }
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape{1, static_cast<int64_t>(audio16k.size())};
    std::array<Ort::Value, 1> inputs{Ort::Value::CreateTensor<float>(mem, const_cast<float*>(audio16k.data()),
                                                                     audio16k.size(), shape.data(), shape.size())};
    auto outputs = session_->run(inputs);
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto oshape = info.GetShape();
    if (oshape.size() != 3 || oshape[1] != frames || oshape[2] != dim_) {
        throw std::runtime_error("content encoder produced unexpected output shape");
    }
    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + static_cast<size_t>(frames) * dim_);
}

void ContentEncoder::prepare_fixed(int samples) {
    fixed_samples_ = samples;
    fixed_frames_ = frames_for_samples(samples);
    if (fixed_frames_ <= 0) throw std::invalid_argument("content encoder window too short");
    fixed_in_.assign(static_cast<size_t>(samples), 0.0f);
    fixed_out_.assign(static_cast<size_t>(fixed_frames_) * dim_, 0.0f);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 2> ishape{1, samples};
    const std::array<int64_t, 3> oshape{1, fixed_frames_, dim_};
    bound_in_.clear();
    bound_out_.clear();
    bound_in_.push_back(Ort::Value::CreateTensor<float>(mem, fixed_in_.data(), fixed_in_.size(), ishape.data(), 2));
    bound_out_.push_back(Ort::Value::CreateTensor<float>(mem, fixed_out_.data(), fixed_out_.size(), oshape.data(), 3));
}

std::span<const float> ContentEncoder::run_fixed() {
    if (bound_in_.empty()) throw std::logic_error("ContentEncoder::run_fixed() before prepare_fixed()");
    session_->run(bound_in_, bound_out_);
    return fixed_out_;
}

}  // namespace xr::rvc
