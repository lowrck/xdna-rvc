#include "inference/session_report.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <spdlog/fmt/fmt.h>
#include <nlohmann/json.hpp>

namespace xr {

int VitisAiReport::nodes_total() const { return nodes_on("all"); }

int VitisAiReport::nodes_on(std::string_view device) const {
    for (const auto& s : device_stats) {
        if (s.name == device) return s.node_count;
    }
    return 0;
}

bool SessionReport::npu_evidence() const {
    if (vitisai_report && vitisai_report->nodes_on("NPU") > 0) return true;
    for (const auto& u : provider_usage) {
        if (u.provider == provider_name(BackendKind::XDNA2) && u.node_count > 0) return true;
    }
    return false;
}

std::optional<double> SessionReport::time_fraction(std::string_view provider) const {
    double total = 0.0, part = 0.0;
    for (const auto& u : provider_usage) {
        total += u.total_us;
        if (u.provider == provider) part += u.total_us;
    }
    if (total <= 0.0) return std::nullopt;
    return part / total;
}

std::string SessionReport::summary() const {
    std::string s = fmt::format("{}: requested {}, running on {}", stage, display_name(requested), display_name(effective));
    if (fell_back) s += fmt::format(" (FALLBACK: {})", fallback_reason);
    if (vitisai_report) {
        s += fmt::format("; VitisAI report: {} nodes, NPU {}, CPU {}", vitisai_report->nodes_total(),
                         vitisai_report->nodes_on("NPU"), vitisai_report->nodes_on("CPU"));
    }
    if (!provider_usage.empty()) {
        s += "; profiled nodes:";
        for (const auto& u : provider_usage) {
            const auto frac = time_fraction(u.provider).value_or(0.0);
            s += fmt::format(" {}={} ({:.0f}% time)", u.provider, u.node_count, frac * 100.0);
        }
    } else if (!evidence_error.empty()) {
        s += "; no execution evidence: " + evidence_error;
    }
    return s;
}

void to_json(nlohmann::json& j, const SessionReport& r) {
    j = nlohmann::json{{"stage", r.stage},
                       {"model", r.model_path.string()},
                       {"requested_backend", std::string(to_string(r.requested))},
                       {"effective_backend", std::string(to_string(r.effective))},
                       {"attempts", r.attempts},
                       {"fell_back", r.fell_back},
                       {"fallback_reason", r.fallback_reason},
                       {"create_ms", r.create_ms},
                       {"used_compiled_cache", r.used_compiled_cache},
                       {"npu_evidence", r.npu_evidence()},
                       {"evidence_error", r.evidence_error},
                       {"profile_file", r.profile_file.string()}};
    auto usage = nlohmann::json::array();
    for (const auto& u : r.provider_usage) {
        usage.push_back({{"provider", u.provider},
                         {"nodes", u.node_count},
                         {"total_us", u.total_us},
                         {"op_types", u.op_types}});
    }
    j["provider_usage"] = usage;
    if (r.vitisai_report) {
        auto stats = nlohmann::json::array();
        for (const auto& s : r.vitisai_report->device_stats) {
            stats.push_back({{"name", s.name}, {"nodes", s.node_count}, {"op_types", s.op_types}});
        }
        j["vitisai_report"] = {{"path", r.vitisai_report->path.string()}, {"device_stats", stats}};
    }
}

std::vector<ProviderUsage> parse_ort_profile(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open ORT profile " + file.string());
    nlohmann::json events;
    try {
        in >> events;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("malformed ORT profile " + file.string() + ": " + e.what());
    }
    if (!events.is_array()) throw std::runtime_error("ORT profile is not a JSON array: " + file.string());

    struct Acc {
        std::set<std::string> nodes;
        std::set<std::string> ops;
        double us = 0.0;
    };
    std::map<std::string, Acc> acc;
    constexpr std::string_view kSuffix = "_kernel_time";
    for (const auto& ev : events) {
        if (!ev.is_object() || ev.value("cat", "") != "Node") continue;
        const std::string name = ev.value("name", "");
        if (name.size() < kSuffix.size() || name.compare(name.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0)
            continue;
        const auto args = ev.find("args");
        if (args == ev.end() || !args->is_object()) continue;
        const std::string provider = args->value("provider", "");
        if (provider.empty()) continue;
        auto& a = acc[provider];
        a.nodes.insert(name.substr(0, name.size() - kSuffix.size()));
        a.ops.insert(args->value("op_name", "?"));
        a.us += ev.value("dur", 0.0);
    }
    std::vector<ProviderUsage> out;
    for (auto& [provider, a] : acc) {
        ProviderUsage u;
        u.provider = provider;
        u.node_count = static_cast<int>(a.nodes.size());
        u.total_us = a.us;
        u.op_types.assign(a.ops.begin(), a.ops.end());
        out.push_back(std::move(u));
    }
    std::sort(out.begin(), out.end(), [](const auto& x, const auto& y) { return x.total_us > y.total_us; });
    return out;
}

VitisAiReport parse_vitisai_report(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open Vitis AI EP report " + file.string());
    nlohmann::json j;
    try {
        in >> j;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("malformed Vitis AI EP report " + file.string() + ": " + e.what());
    }
    VitisAiReport report;
    report.path = file;
    const auto stats = j.find("deviceStat");
    if (stats == j.end() || !stats->is_array()) {
        throw std::runtime_error("Vitis AI EP report has no 'deviceStat' array: " + file.string());
    }
    for (const auto& s : *stats) {
        VitisAiDeviceStat d;
        d.name = s.value("name", "");
        d.node_count = s.value("nodeNum", 0);
        if (const auto ops = s.find("supportedOpType"); ops != s.end() && ops->is_array()) {
            for (const auto& op : *ops) {
                if (op.is_string()) d.op_types.push_back(op.get<std::string>());
            }
        }
        report.device_stats.push_back(std::move(d));
    }
    return report;
}

}  // namespace xr
