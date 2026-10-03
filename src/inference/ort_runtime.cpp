#include "inference/ort_runtime.h"

#include <algorithm>

#include "util/log.h"

#ifndef XDNA_RVC_ORT_FLAVOR
#define XDNA_RVC_ORT_FLAVOR "unknown"
#endif

namespace xr {

OrtRuntime::OrtRuntime(OrtLoggingLevel level)
    : env_(level, "xdna-rvc"),
      version_(Ort::GetVersionString()),
      build_flavor_(XDNA_RVC_ORT_FLAVOR),
      providers_(Ort::GetAvailableProviders()),
      npu_(detect_npu()) {
    std::string list;
    for (const auto& p : providers_) list += (list.empty() ? "" : ", ") + p;
    XR_LOG_INFO("ONNX Runtime {} ({} package); available providers: {}", version_, build_flavor_, list);
    XR_LOG_INFO("NPU: {} - {}", to_string(npu_.kind), npu_.detail);
}

bool OrtRuntime::has_provider(std::string_view name) const {
    return std::find(providers_.begin(), providers_.end(), name) != providers_.end();
}

}  // namespace xr
