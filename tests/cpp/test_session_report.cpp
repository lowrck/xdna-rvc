#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include "inference/session_report.h"

using namespace xr;

namespace {

std::filesystem::path write_temp(const std::string& name, const std::string& content) {
    const auto dir = std::filesystem::temp_directory_path() / "xdna_rvc_tests";
    std::filesystem::create_directories(dir);
    const auto p = dir / name;
    std::ofstream(p) << content;
    return p;
}

}  // namespace

TEST_CASE("ORT profile parsing aggregates kernel events per provider") {
    // Shape of ONNX Runtime's chrome-trace profiler output.
    const auto p = write_temp("profile.json", R"([
      {"cat":"Session","name":"model_loading_uri","dur":100,"args":{}},
      {"cat":"Node","name":"conv1_fence_before","dur":0,"args":{"op_name":"Conv","provider":"CPUExecutionProvider"}},
      {"cat":"Node","name":"conv1_kernel_time","dur":50,"args":{"op_name":"Conv","provider":"CPUExecutionProvider"}},
      {"cat":"Node","name":"conv1_kernel_time","dur":30,"args":{"op_name":"Conv","provider":"CPUExecutionProvider"}},
      {"cat":"Node","name":"gru_kernel_time","dur":20,"args":{"op_name":"GRU","provider":"CPUExecutionProvider"}},
      {"cat":"Node","name":"vaiml_par_0_kernel_time","dur":400,"args":{"op_name":"vaiml_par_0","provider":"VitisAIExecutionProvider"}}
    ])");
    const auto usage = parse_ort_profile(p);
    REQUIRE(usage.size() == 2);
    CHECK(usage[0].provider == "VitisAIExecutionProvider");  // sorted by time
    CHECK(usage[0].node_count == 1);
    CHECK(usage[0].total_us == doctest::Approx(400));
    CHECK(usage[1].provider == "CPUExecutionProvider");
    CHECK(usage[1].node_count == 2);  // conv1 counted once even though it ran twice
    CHECK(usage[1].total_us == doctest::Approx(100));

    SessionReport r;
    r.provider_usage = usage;
    CHECK(r.npu_evidence());
    CHECK(r.time_fraction("VitisAIExecutionProvider").value() == doctest::Approx(0.8));
}

TEST_CASE("profile without NPU nodes is not NPU evidence") {
    SessionReport r;
    r.provider_usage.push_back({"CPUExecutionProvider", 10, 100.0, {"Conv"}});
    CHECK_FALSE(r.npu_evidence());
    CHECK_FALSE(SessionReport{}.time_fraction("CPUExecutionProvider").has_value());
}

TEST_CASE("Vitis AI EP report parsing (format from Ryzen AI 1.8 docs)") {
    const auto p = write_temp("vitisai_ep_report.json", R"({
      "deviceStat": [
        {"name": "all", "nodeNum": 400, "supportedOpType": ["::Add", "::Conv"]},
        {"name": "CPU", "nodeNum": 2, "supportedOpType": ["::DequantizeLinear", "::QuantizeLinear"]},
        {"name": "NPU", "nodeNum": 398, "supportedOpType": ["::Add", "::Conv"]}
      ]
    })");
    const auto rep = parse_vitisai_report(p);
    CHECK(rep.nodes_total() == 400);
    CHECK(rep.nodes_on("NPU") == 398);
    CHECK(rep.nodes_on("CPU") == 2);
    CHECK(rep.nodes_on("GPU") == 0);

    SessionReport r;
    r.vitisai_report = rep;
    CHECK(r.npu_evidence());
}

TEST_CASE("malformed evidence files raise errors") {
    CHECK_THROWS(parse_ort_profile(write_temp("bad.json", "{not json")));
    CHECK_THROWS(parse_vitisai_report(write_temp("bad2.json", R"({"other": 1})")));
    CHECK_THROWS(parse_ort_profile("/nonexistent/file.json"));
}
