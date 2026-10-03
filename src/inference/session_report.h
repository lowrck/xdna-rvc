#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "inference/backend_kind.h"

namespace xr {

// Per-provider execution evidence, aggregated from an ONNX Runtime profile.
struct ProviderUsage {
    std::string provider;          // e.g. "VitisAIExecutionProvider"
    int node_count = 0;            // distinct graph nodes executed by this provider
    double total_us = 0.0;         // summed kernel time in the profiled run(s)
    std::vector<std::string> op_types;  // distinct op types
};

// One entry of the Vitis AI EP operator assignment report ("deviceStat").
struct VitisAiDeviceStat {
    std::string name;  // "all", "CPU", "NPU", ...
    int node_count = 0;
    std::vector<std::string> op_types;
};

struct VitisAiReport {
    std::filesystem::path path;
    std::vector<VitisAiDeviceStat> device_stats;
    int nodes_total() const;
    int nodes_on(std::string_view device) const;
};

// Everything we know about how one model was placed on hardware.
struct SessionReport {
    std::string stage;                      // "content_encoder", "rmvpe", "generator", ...
    std::filesystem::path model_path;
    BackendKind requested = BackendKind::Auto;
    BackendKind effective = BackendKind::CPU;
    std::vector<std::string> attempts;      // one line per backend tried, with outcome
    bool fell_back = false;                 // effective != requested (or != first Auto choice)
    std::string fallback_reason;
    double create_ms = 0.0;
    bool used_compiled_cache = false;       // XDNA: cache entry existed before session creation
    std::filesystem::path cache_dir;

    std::optional<VitisAiReport> vitisai_report;
    std::vector<ProviderUsage> provider_usage;   // from the evidence-probe ORT profile
    std::filesystem::path profile_file;
    std::string evidence_error;                  // why evidence could not be collected

    // True only if there is positive evidence that graph nodes executed on the NPU.
    bool npu_evidence() const;
    // Fraction of profiled node time spent in `provider` (0..1), or nullopt without a profile.
    std::optional<double> time_fraction(std::string_view provider) const;
    std::string summary() const;
};

void to_json(nlohmann::json& j, const SessionReport& r);

// Parses an ONNX Runtime profiler JSON file (chrome trace format) and aggregates node
// kernel events per execution provider. Throws std::runtime_error on malformed input.
std::vector<ProviderUsage> parse_ort_profile(const std::filesystem::path& file);
// Parses a Vitis AI EP report (XLNX_ONNX_EP_REPORT_FILE). Throws on malformed input.
VitisAiReport parse_vitisai_report(const std::filesystem::path& file);

}  // namespace xr
