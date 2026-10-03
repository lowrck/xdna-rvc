#include "inference/backends.h"

#include "inference/ort_runtime.h"

namespace xr {

std::unique_ptr<IInferenceBackend> make_backend(BackendKind kind) {
    switch (kind) {
        case BackendKind::CPU: return std::make_unique<CpuBackend>();
        case BackendKind::DirectML: return std::make_unique<DirectMLBackend>();
        case BackendKind::XDNA2: return std::make_unique<Xdna2Backend>();
        case BackendKind::CUDA:
        case BackendKind::ROCm:
        case BackendKind::Auto: return nullptr;
    }
    return nullptr;
}

std::vector<BackendStatus> probe_backends(const OrtRuntime& runtime) {
    std::vector<BackendStatus> out;
    for (BackendKind k : {BackendKind::XDNA2, BackendKind::DirectML, BackendKind::CPU, BackendKind::CUDA,
                          BackendKind::ROCm}) {
        auto backend = make_backend(k);
        if (backend) {
            out.push_back({k, backend->availability(runtime)});
        } else {
            const bool in_ort = runtime.has_provider(provider_name(k));
            out.push_back({k, {false, std::string("no backend implementation in this build") +
                                          (in_ort ? " (provider is present in ORT)" : "")}});
        }
    }
    return out;
}

}  // namespace xr
