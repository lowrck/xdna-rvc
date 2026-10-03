#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace xr {

// Execution backends. Each maps to one ONNX Runtime execution provider.
// CUDA and ROCm are reserved: the enum values and provider names exist so the
// configuration format and UI are stable, but this build provides no backend
// implementation for them (make_backend() reports them as not implemented).
enum class BackendKind { Auto, XDNA2, DirectML, CPU, CUDA, ROCm };

std::string_view to_string(BackendKind kind);
std::string_view display_name(BackendKind kind);
// ONNX Runtime provider name, e.g. "VitisAIExecutionProvider". Empty for Auto.
std::string_view provider_name(BackendKind kind);
// Accepts auto|xdna2|npu|vitisai|directml|dml|cpu|cuda|rocm. Throws std::invalid_argument.
BackendKind parse_backend(std::string_view text);

// Order used by Automatic mode.
const std::vector<BackendKind>& automatic_preference();

}  // namespace xr
