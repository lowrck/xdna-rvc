#include "inference/ort_runtime.h"

#include <algorithm>

#include "util/log.h"
#include "util/thread_priority.h"

#include <thread>

#ifndef XDNA_RVC_ORT_FLAVOR
#define XDNA_RVC_ORT_FLAVOR "unknown"
#endif

namespace xr {

namespace {

// ORT global thread pool threads created by us so they can run at raised priority.
struct PoolThread {
    std::thread thread;
};

OrtCustomThreadHandle create_pool_thread(void*, OrtThreadWorkerFn fn, void* param) {
    auto* t = new PoolThread;
    t->thread = std::thread([fn, param] {
        set_current_thread_priority(ThreadPriority::High);
        fn(param);
    });
    return reinterpret_cast<OrtCustomThreadHandle>(t);
}

void join_pool_thread(OrtCustomThreadHandle h) {
    auto* t = reinterpret_cast<PoolThread*>(const_cast<OrtCustomHandleType*>(h));
    if (t->thread.joinable()) t->thread.join();
    delete t;
}

}  // namespace

Ort::Env OrtRuntime::make_env(OrtLoggingLevel level, int global_threads, bool raise_pool_priority) {
    if (global_threads <= 0) return Ort::Env(level, "xdna-rvc");
    Ort::ThreadingOptions to;
    to.SetGlobalIntraOpNumThreads(global_threads);
    to.SetGlobalInterOpNumThreads(1);
    if (raise_pool_priority) {
        to.SetGlobalCustomCreateThreadFn(create_pool_thread);
        to.SetGlobalCustomJoinThreadFn(join_pool_thread);
    }
    return Ort::Env(to, level, "xdna-rvc");
}

OrtRuntime::OrtRuntime(OrtLoggingLevel level, int global_threads, bool raise_pool_priority)
    : global_threads_(std::max(0, global_threads)),
      env_(make_env(level, global_threads, raise_pool_priority)),
      version_(Ort::GetVersionString()),
      build_flavor_(XDNA_RVC_ORT_FLAVOR),
      providers_(Ort::GetAvailableProviders()),
      npu_(detect_npu()) {
    std::string list;
    for (const auto& p : providers_) list += (list.empty() ? "" : ", ") + p;
    XR_LOG_INFO("ONNX Runtime {} ({} package); available providers: {}", version_, build_flavor_, list);
    XR_LOG_INFO("NPU: {} - {}", to_string(npu_.kind), npu_.detail);
    if (global_threads_ > 0) {
        XR_LOG_INFO("ONNX Runtime global intra-op thread pool: {} threads{}", global_threads_,
                    raise_pool_priority ? ", raised priority" : "");
    }
}

bool OrtRuntime::has_provider(std::string_view name) const {
    return std::find(providers_.begin(), providers_.end(), name) != providers_.end();
}

}  // namespace xr
