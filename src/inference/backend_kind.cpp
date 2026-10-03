#include "inference/backend_kind.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace xr {

std::string_view to_string(BackendKind kind) {
    switch (kind) {
        case BackendKind::Auto: return "auto";
        case BackendKind::XDNA2: return "xdna2";
        case BackendKind::DirectML: return "directml";
        case BackendKind::CPU: return "cpu";
        case BackendKind::CUDA: return "cuda";
        case BackendKind::ROCm: return "rocm";
    }
    return "auto";
}

std::string_view display_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Auto: return "Automatic";
        case BackendKind::XDNA2: return "XDNA 2";
        case BackendKind::DirectML: return "DirectML";
        case BackendKind::CPU: return "CPU";
        case BackendKind::CUDA: return "CUDA";
        case BackendKind::ROCm: return "ROCm";
    }
    return "Automatic";
}

std::string_view provider_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Auto: return "";
        case BackendKind::XDNA2: return "VitisAIExecutionProvider";
        case BackendKind::DirectML: return "DmlExecutionProvider";
        case BackendKind::CPU: return "CPUExecutionProvider";
        case BackendKind::CUDA: return "CUDAExecutionProvider";
        case BackendKind::ROCm: return "MIGraphXExecutionProvider";
    }
    return "";
}

BackendKind parse_backend(std::string_view text) {
    std::string s(text);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "auto" || s == "automatic") return BackendKind::Auto;
    if (s == "xdna2" || s == "xdna" || s == "npu" || s == "vitisai") return BackendKind::XDNA2;
    if (s == "directml" || s == "dml") return BackendKind::DirectML;
    if (s == "cpu") return BackendKind::CPU;
    if (s == "cuda") return BackendKind::CUDA;
    if (s == "rocm" || s == "migraphx") return BackendKind::ROCm;
    throw std::invalid_argument("unknown backend '" + std::string(text) + "' (expected auto|xdna2|directml|cpu)");
}

const std::vector<BackendKind>& automatic_preference() {
    static const std::vector<BackendKind> order = {BackendKind::XDNA2, BackendKind::DirectML, BackendKind::CPU};
    return order;
}

}  // namespace xr
