#include "rvc/generator.h"

#include <fmt/format.h>

#include "util/error.h"

namespace xr::rvc {

namespace {

template <typename T>
Ort::Value bind_tensor(const OrtMemoryInfo* mem, std::vector<T>& buf, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<T>(mem, buf.data(), buf.size(), shape.data(), shape.size());
}

size_t count(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
}

}  // namespace

Generator::Generator(OrtRuntime& runtime, SessionRequest request, OpenOptions options, const ModelInfo& model,
                     const GeneratorVariant& variant)
    : uses_f0_(model.uses_f0) {
    if (request.stage.empty()) request.stage = "generator";
    request.model_path = variant.path;
    session_ = ModelSession::open(runtime, request, options);

    const std::vector<std::string> expected =
        uses_f0_ ? std::vector<std::string>{"phone", "pitch", "pitchf", "sid", "rnd", "noise"}
                 : std::vector<std::string>{"phone", "sid", "rnd"};
    const auto& ins = session_->inputs();
    if (ins.size() != expected.size()) {
        throw UserError(fmt::format("generator {} has {} inputs, expected {} for a {} model", variant.path.string(),
                                    ins.size(), expected.size(), uses_f0_ ? "F0" : "no-F0"),
                        "Re-export it with tools/convert_rvc.py.");
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (ins[i].name != expected[i] || !ins[i].is_static()) {
            throw UserError(fmt::format("generator input {} is '{}' ({}), expected static '{}'", i, ins[i].name,
                                        ins[i].to_string(), expected[i]));
        }
    }
    const auto& phone_shape = session_->input("phone").shape;
    frames_ = static_cast<int>(phone_shape[1]);
    if (phone_shape[2] != model.feature_dim) {
        throw UserError(fmt::format("generator phone dim {} != model feature dim {}", phone_shape[2],
                                    model.feature_dim));
    }
    const auto& rnd_shape = session_->input("rnd").shape;
    flow_frames_ = static_cast<int>(rnd_shape[2]);
    const auto& out = session_->outputs().at(0);
    if (!out.is_static() || out.shape.size() != 2) {
        throw UserError("generator output must be static [1, samples], got " + out.to_string());
    }
    output_samples_ = static_cast<int>(out.shape[1]);
    if (frames_ != variant.frames || output_samples_ != variant.output_samples) {
        throw UserError(fmt::format("generator {} does not match model.json (frames {} vs {}, samples {} vs {})",
                                    variant.path.string(), frames_, variant.frames, output_samples_,
                                    variant.output_samples));
    }

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    phone_.assign(count(phone_shape), 0.0f);
    rnd_.assign(count(rnd_shape), 0.0f);
    sid_.assign(1, 0);
    audio_.assign(static_cast<size_t>(output_samples_), 0.0f);
    in_.push_back(bind_tensor(mem, phone_, phone_shape));
    if (uses_f0_) {
        pitch_.assign(static_cast<size_t>(frames_), 1);
        pitchf_.assign(static_cast<size_t>(frames_), 0.0f);
        noise_.assign(count(session_->input("noise").shape), 0.0f);
        in_.push_back(bind_tensor(mem, pitch_, session_->input("pitch").shape));
        in_.push_back(bind_tensor(mem, pitchf_, session_->input("pitchf").shape));
        in_.push_back(bind_tensor(mem, sid_, session_->input("sid").shape));
        in_.push_back(bind_tensor(mem, rnd_, rnd_shape));
        in_.push_back(bind_tensor(mem, noise_, session_->input("noise").shape));
    } else {
        in_.push_back(bind_tensor(mem, sid_, session_->input("sid").shape));
        in_.push_back(bind_tensor(mem, rnd_, rnd_shape));
    }
    out_.push_back(bind_tensor(mem, audio_, out.shape));
}

std::span<const float> Generator::run() {
    session_->run(in_, out_);
    return audio_;
}

}  // namespace xr::rvc
