#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "inference/backend_kind.h"
#include "inference/session_report.h"

namespace xr {

class OrtRuntime;

enum class ModelPrecision { Float, Int8 };

// Everything needed to open one model for one pipeline stage.
struct SessionRequest {
    std::string stage;                    // used for logs, cache keys and reports
    std::filesystem::path model_path;
    ModelPrecision precision = ModelPrecision::Float;
    int intra_op_threads = 0;             // 0 = ORT default
    // XDNA compile cache. The cache key should change whenever the model bytes change.
    std::filesystem::path cache_dir;
    std::string cache_key;
    // BF16 compile configuration (vaiml_config). Must be the file the model was precompiled with.
    std::filesystem::path xdna_config_file;
    // When true the session fails instead of letting ORT assign unsupported nodes to the CPU EP
    // ("session.disable_cpu_ep_fallback"). Useful to prove a model is fully NPU-resident.
    bool disallow_cpu_ep_fallback = false;
    // Execution evidence: create with ORT profiling, run once, parse the profile.
    bool collect_evidence = true;
    std::filesystem::path diagnostics_dir = "diagnostics";
    // Keep ORT profiling enabled for the whole session (in addition to the probe).
    bool keep_profiling = false;
    // AMD AI Analyzer artifacts (BF16 models only, per Ryzen AI 1.8 docs).
    bool ai_analyzer = false;
    int directml_device_id = 0;
    // Pins symbolic input dimensions (e.g. {"samples", 12000}) so the session runs with
    // static shapes. For XDNA 2 a static copy of the model is written to `static_model_dir`
    // and compiled instead, because the NPU compiler requires static shapes.
    std::map<std::string, int64_t> free_dims;
    std::filesystem::path static_model_dir = "cache/static_models";
    OrtLoggingLevel session_log_level = ORT_LOGGING_LEVEL_WARNING;
};

struct BackendAvailability {
    bool available = false;
    std::string reason;  // why available or not; always filled
};

// Configures ONNX Runtime session options for one execution provider and collects
// provider-specific evidence afterwards. Implementations: CpuBackend, DirectMLBackend,
// Xdna2Backend.
class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    virtual BackendKind kind() const = 0;
    virtual BackendAvailability availability(const OrtRuntime& runtime) const = 0;
    // Appends the execution provider. Throws Ort::Exception / UserError on failure.
    virtual void configure(Ort::SessionOptions& options, const SessionRequest& request) const = 0;
    // Called before session creation (e.g. to note whether a compiled cache exists).
    virtual void before_create(const SessionRequest& request, SessionReport& report) const {}
    // Called after successful session creation to gather compile-time evidence.
    virtual void after_create(const SessionRequest& request, SessionReport& report) const {}
};

// Returns nullptr for kinds without an implementation in this build (CUDA, ROCm, Auto).
std::unique_ptr<IInferenceBackend> make_backend(BackendKind kind);

// Availability of every backend kind, in display order, for diagnostics.
struct BackendStatus {
    BackendKind kind;
    BackendAvailability availability;
};
std::vector<BackendStatus> probe_backends(const OrtRuntime& runtime);

}  // namespace xr
