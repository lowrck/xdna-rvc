#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "inference/npu_detect.h"

namespace xr {

// Process-level ONNX Runtime state: the Ort::Env plus what the loaded runtime can
// do. Create exactly one, early in main(), and pass it by reference. It is not a
// global so that tests can construct their own.
class OrtRuntime {
public:
    explicit OrtRuntime(OrtLoggingLevel level = ORT_LOGGING_LEVEL_WARNING);

    Ort::Env& env() { return env_; }
    const std::string& version() const { return version_; }
    // Compile-time flavor of the ORT package we linked (cpu/directml/ryzenai/custom).
    const std::string& build_flavor() const { return build_flavor_; }
    const std::vector<std::string>& available_providers() const { return providers_; }
    bool has_provider(std::string_view name) const;
    const NpuInfo& npu() const { return npu_; }

private:
    Ort::Env env_;
    std::string version_;
    std::string build_flavor_;
    std::vector<std::string> providers_;
    NpuInfo npu_;
};

}  // namespace xr
