#include "inference/backends.h"

#include "inference/ort_runtime.h"

namespace xr {

BackendAvailability CpuBackend::availability(const OrtRuntime& runtime) const {
    if (runtime.has_provider(provider_name(BackendKind::CPU))) {
        return {true, "CPUExecutionProvider is always available"};
    }
    return {false, "CPUExecutionProvider missing from this ONNX Runtime build (unexpected)"};
}

void CpuBackend::configure(Ort::SessionOptions& options, const SessionRequest& request) const {
    // The CPU EP is implicit; nothing to append. Thread count is set by the session wrapper.
}

}  // namespace xr
