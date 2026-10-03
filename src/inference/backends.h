#pragma once

#include "inference/inference_backend.h"

namespace xr {

class CpuBackend final : public IInferenceBackend {
public:
    BackendKind kind() const override { return BackendKind::CPU; }
    BackendAvailability availability(const OrtRuntime& runtime) const override;
    void configure(Ort::SessionOptions& options, const SessionRequest& request) const override;
};

// DirectML (Windows, D3D12 GPU). The DML entry point is resolved from the loaded
// onnxruntime.dll at runtime so the build does not depend on d3d12/DirectML headers
// and works with any ORT package that contains the DML EP (Microsoft DirectML
// package or AMD's Ryzen AI package).
class DirectMLBackend final : public IInferenceBackend {
public:
    BackendKind kind() const override { return BackendKind::DirectML; }
    BackendAvailability availability(const OrtRuntime& runtime) const override;
    void configure(Ort::SessionOptions& options, const SessionRequest& request) const override;
};

// AMD Ryzen AI NPU through the VitisAI execution provider (Ryzen AI Software 1.8).
//
// Provider options used (Ryzen AI 1.8 "Vitis AI EP Options Reference Guide"):
//   cache_dir, cache_key                 compiled model cache location
//   enable_cache_file_io_in_mem = "0"    persist compiled output to disk; also required
//                                        for the operator assignment report
//   config_file                          BF16 (vaiml) compile configuration, float models
//   target = "X2"                        INT8 models on STX/KRK
//   ai_analyzer_visualization/profiling  optional AI Analyzer artifacts
// Environment:
//   XLNX_ONNX_EP_REPORT_FILE             name of the CPU/NPU assignment report written
//                                        into the cache directory
class Xdna2Backend final : public IInferenceBackend {
public:
    static constexpr const char* kReportFileName = "vitisai_ep_report.json";

    BackendKind kind() const override { return BackendKind::XDNA2; }
    BackendAvailability availability(const OrtRuntime& runtime) const override;
    void configure(Ort::SessionOptions& options, const SessionRequest& request) const override;
    void before_create(const SessionRequest& request, SessionReport& report) const override;
    void after_create(const SessionRequest& request, SessionReport& report) const override;
};

}  // namespace xr
