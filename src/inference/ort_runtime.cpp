#include "inference/ort_runtime.h"

#include <algorithm>

#include "util/log.h"

#ifndef XDNA_RVC_ORT_FLAVOR
#define XDNA_RVC_ORT_FLAVOR "unknown"
#endif

namespace xr {

Ort::Env OrtRuntime::make_env(OrtLoggingLevel level, int global_threads) {
    if (global_threads <= 0) return Ort::Env(level, "xdna-rvc");
    Ort::ThreadingOptions to;
    to.SetGlobalIntraOpNumThreads(global_threads);
    to.SetGlobalInterOpNumThreads(1);
    return Ort::Env(to, level, "xdna-rvc");
}

OrtRuntime::OrtRuntime(OrtLoggingLevel level, int global_threads)
    : global_threads_(std::max(0, global_threads)),
      env_(make_env(level, global_threads)),
      version_(Ort::GetVersionString()),
      build_flavor_(XDNA_RVC_ORT_FLAVOR),
      providers_(Ort::GetAvailableProviders()),
      npu_(detect_npu()) {
    std::string list;
    for (const auto& p : providers_) list += (list.empty() ? "" : ", ") + p;
    XR_LOG_INFO("ONNX Runtime {} ({} package); available providers: {}", version_, build_flavor_, list);
    XR_LOG_INFO("NPU: {} - {}", to_string(npu_.kind), npu_.detail);
    if (global_threads_ > 0) XR_LOG_INFO("ONNX Runtime global intra-op thread pool: {} threads", global_threads_);
}

bool OrtRuntime::has_provider(std::string_view name) const {
    return std::find(providers_.begin(), providers_.end(), name) != providers_.end();
}

}  // namespace xr
