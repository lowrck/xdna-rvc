#include "inference/backends.h"

#include "inference/ort_runtime.h"
#include "util/error.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace xr {

namespace {

#if defined(_WIN32)
// Signature from dml_provider_factory.h:
//   ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_DML, OrtSessionOptions* options, int device_id);
using AppendDmlFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);

AppendDmlFn resolve_append_dml() {
    HMODULE ort = GetModuleHandleW(L"onnxruntime.dll");
    if (!ort) return nullptr;
    return reinterpret_cast<AppendDmlFn>(GetProcAddress(ort, "OrtSessionOptionsAppendExecutionProvider_DML"));
}
#endif

}  // namespace

BackendAvailability DirectMLBackend::availability(const OrtRuntime& runtime) const {
#if defined(_WIN32)
    if (!runtime.has_provider(provider_name(BackendKind::DirectML))) {
        return {false, "DmlExecutionProvider is not in this ONNX Runtime build (built with flavor '" +
                           runtime.build_flavor() + "'). Rebuild with -DXDNA_RVC_ORT_FLAVOR=directml or ryzenai."};
    }
    if (!resolve_append_dml()) {
        return {false, "onnxruntime.dll does not export OrtSessionOptionsAppendExecutionProvider_DML"};
    }
    return {true, "DmlExecutionProvider present"};
#else
    return {false, "DirectML is only available on Windows"};
#endif
}

void DirectMLBackend::configure(Ort::SessionOptions& options, const SessionRequest& request) const {
#if defined(_WIN32)
    auto fn = resolve_append_dml();
    if (!fn) throw UserError("DirectML entry point not found in onnxruntime.dll");
    // Required by the DirectML EP: no memory pattern optimisation, sequential execution.
    options.DisableMemPattern();
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    Ort::ThrowOnError(fn(options, request.directml_device_id));
#else
    throw UserError("DirectML backend requested on a non-Windows platform",
                    "Use --backend cpu on this platform.");
#endif
}

}  // namespace xr
