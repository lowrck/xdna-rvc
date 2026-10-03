#include "inference/backends.h"

#include <cstdlib>
#include <unordered_map>

#include "inference/ort_runtime.h"
#include "util/error.h"
#include "util/log.h"

namespace xr {

namespace {

void set_env(const char* name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

}  // namespace

BackendAvailability Xdna2Backend::availability(const OrtRuntime& runtime) const {
    const NpuInfo& npu = runtime.npu();
    if (!runtime.has_provider(provider_name(BackendKind::XDNA2))) {
        std::string reason = "VitisAIExecutionProvider is not in this ONNX Runtime build (flavor '" +
                             runtime.build_flavor() + "').";
#if defined(_WIN32)
        reason += " Install Ryzen AI Software 1.8+ and rebuild with -DXDNA_RVC_ORT_FLAVOR=ryzenai "
                  "(see docs/amd_xdna_setup.md).";
#else
        reason += " The Ryzen AI VitisAI EP is distributed for Windows.";
#endif
        if (npu.present()) reason += std::string(" An NPU was detected: ") + to_string(npu.kind) + ".";
        return {false, reason};
    }
    if (!npu.present()) {
        return {false, "VitisAIExecutionProvider is present but no AMD NPU was detected: " + npu.detail};
    }
    if (!is_xdna2(npu.kind)) {
        return {false, std::string("Detected ") + to_string(npu.kind) +
                           "; this backend targets XDNA 2 (STX/KRK). BF16 models are unsupported on PHX/HPT."};
    }
    if (npu.driver_status == NpuDriverStatus::TooOld || npu.driver_status == NpuDriverStatus::NotInstalled) {
        return {false, npu.detail};
    }
    return {true, std::string(to_string(npu.kind)) + " with VitisAIExecutionProvider; " + npu.detail};
}

void Xdna2Backend::configure(Ort::SessionOptions& options, const SessionRequest& request) const {
    std::unordered_map<std::string, std::string> vai;
    if (!request.cache_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(request.cache_dir, ec);
        vai["cache_dir"] = std::filesystem::absolute(request.cache_dir).string();
    }
    if (!request.cache_key.empty()) vai["cache_key"] = request.cache_key;
    // Persist the compiled model and allow the operator assignment report to be written.
    vai["enable_cache_file_io_in_mem"] = "0";

    if (request.precision == ModelPrecision::Int8) {
        vai["target"] = "X2";  // STX/KRK INT8 backend. xclbin must not be set on STX/KRK.
    } else if (!request.xdna_config_file.empty()) {
        if (!std::filesystem::exists(request.xdna_config_file)) {
            throw UserError("XDNA BF16 compile config not found: " + request.xdna_config_file.string(),
                            "Re-run tools/compile_xdna.py for this model, or pass the config it used.");
        }
        vai["config_file"] = std::filesystem::absolute(request.xdna_config_file).string();
    }
    if (request.ai_analyzer) {
        vai["ai_analyzer_visualization"] = "1";
        vai["ai_analyzer_profiling"] = "1";
    }

    // The EP writes this report into the cache directory during compilation.
    set_env("XLNX_ONNX_EP_REPORT_FILE", kReportFileName);

    std::string desc;
    for (const auto& [k, v] : vai) desc += (desc.empty() ? "" : ", ") + k + "=" + v;
    XR_LOG_INFO("[{}] VitisAI EP options: {}", request.stage, desc);
    options.AppendExecutionProvider_VitisAI(vai);
}

void Xdna2Backend::before_create(const SessionRequest& request, SessionReport& report) const {
    report.cache_dir = request.cache_dir;
    if (!request.cache_dir.empty() && !request.cache_key.empty()) {
        std::error_code ec;
        report.used_compiled_cache = std::filesystem::is_directory(request.cache_dir / request.cache_key, ec) &&
                                     !std::filesystem::is_empty(request.cache_dir / request.cache_key, ec);
    }
    XR_LOG_INFO("[{}] XDNA compiled cache {} ({})", request.stage,
                report.used_compiled_cache ? "found, expecting fast load" : "not found, the model will be compiled",
                (request.cache_dir / request.cache_key).string());
}

void Xdna2Backend::after_create(const SessionRequest& request, SessionReport& report) const {
    // Ryzen AI docs: "the report file is automatically generated in the cache directory".
    // Check the key subdirectory first, then the cache root.
    const std::filesystem::path candidates[] = {request.cache_dir / request.cache_key / kReportFileName,
                                                request.cache_dir / kReportFileName,
                                                std::filesystem::path(kReportFileName)};
    for (const auto& p : candidates) {
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) continue;
        try {
            report.vitisai_report = parse_vitisai_report(p);
            XR_LOG_INFO("[{}] VitisAI EP report {}: {} nodes total, {} on NPU, {} on CPU", request.stage, p.string(),
                        report.vitisai_report->nodes_total(), report.vitisai_report->nodes_on("NPU"),
                        report.vitisai_report->nodes_on("CPU"));
        } catch (const std::exception& e) {
            XR_LOG_WARN("[{}] could not parse VitisAI EP report {}: {}", request.stage, p.string(), e.what());
        }
        return;
    }
    XR_LOG_WARN("[{}] no VitisAI EP operator assignment report found under {}. CPU/NPU node placement will be "
                "inferred from the ORT profile only.",
                request.stage, request.cache_dir.string());
}

}  // namespace xr
