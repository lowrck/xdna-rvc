#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "inference/backend_kind.h"
#include "inference/inference_backend.h"
#include "inference/session_report.h"

namespace xr {

class OrtRuntime;

struct TensorInfo {
    std::string name;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64_t> shape;  // -1 for dynamic dimensions
    std::vector<std::string> dim_names;  // symbolic names of dimensions ("" if none)
    bool is_static() const;
    std::string to_string() const;
};

const char* element_type_name(ONNXTensorElementDataType type);

struct OpenOptions {
    BackendKind backend = BackendKind::Auto;
    // When an explicitly requested backend fails, retry on CPU instead of failing.
    // The fallback is always logged and recorded in the SessionReport.
    bool allow_cpu_fallback = false;
    // Concrete shapes for dynamic inputs, used by the evidence probe run.
    std::map<std::string, std::vector<int64_t>> probe_shapes;
};

// RAII wrapper around one Ort::Session plus the provenance of how it was created.
class ModelSession {
public:
    static std::unique_ptr<ModelSession> open(OrtRuntime& runtime, const SessionRequest& request,
                                            const OpenOptions& options);

    const std::vector<TensorInfo>& inputs() const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }
    const TensorInfo& input(std::string_view name) const;
    const TensorInfo& output(std::string_view name) const;
    bool has_input(std::string_view name) const;
    const SessionReport& report() const { return report_; }
    BackendKind backend() const { return report_.effective; }

    // Runs with caller-provided (pre-allocated) outputs: no output allocation. The
    // order of `in`/`out` must match inputs()/outputs().
    void run(std::span<const Ort::Value> in, std::span<Ort::Value> out);
    // Allocating convenience variant.
    std::vector<Ort::Value> run(std::span<const Ort::Value> in);

    // Ends a session-long profile (keep_profiling) and returns the file path, or empty.
    std::string end_profiling();

private:
    ModelSession(Ort::Session session, SessionReport report, bool profiling_active);
    void collect_evidence(const OpenOptions& options, const SessionRequest& request);
    std::map<std::string, int64_t> free_dims_;

    Ort::Session session_;
    SessionReport report_;
    bool profiling_active_ = false;
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    Ort::RunOptions run_options_;
};

// Writes (or reuses from `cache_dir`) a copy of `src` whose symbolic input dimensions are
// fixed to `dims` (graph otherwise unchanged).
std::filesystem::path materialize_static_model(OrtRuntime& runtime, const std::filesystem::path& src,
                                              const std::map<std::string, int64_t>& dims,
                                              const std::filesystem::path& cache_dir);

// Creates a zero-filled tensor of the given type/shape owning its memory, for probes and tests.
Ort::Value make_zero_tensor(ONNXTensorElementDataType type, const std::vector<int64_t>& shape);

}  // namespace xr
