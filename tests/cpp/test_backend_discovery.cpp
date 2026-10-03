#include <doctest/doctest.h>

#include "inference/backend_kind.h"
#include "inference/inference_backend.h"
#include "inference/npu_detect.h"
#include "inference/ort_runtime.h"

using namespace xr;

TEST_CASE("backend names round-trip") {
    for (BackendKind k : {BackendKind::Auto, BackendKind::XDNA2, BackendKind::DirectML, BackendKind::CPU,
                          BackendKind::CUDA, BackendKind::ROCm}) {
        CHECK(parse_backend(to_string(k)) == k);
    }
    CHECK(parse_backend("NPU") == BackendKind::XDNA2);
    CHECK(parse_backend("dml") == BackendKind::DirectML);
    CHECK_THROWS_AS(parse_backend("tpu"), std::invalid_argument);
    CHECK(provider_name(BackendKind::XDNA2) == "VitisAIExecutionProvider");
    CHECK(provider_name(BackendKind::DirectML) == "DmlExecutionProvider");
    CHECK(automatic_preference().front() == BackendKind::XDNA2);
    CHECK(automatic_preference().back() == BackendKind::CPU);
}

TEST_CASE("NPU PCI classification follows the Ryzen AI APU table") {
    CHECK(classify_npu(0x1022, 0x1502, 0x00) == NpuKind::PhoenixHawkPoint);
    CHECK(classify_npu(0x1022, 0x17F0, 0x00) == NpuKind::Strix);
    CHECK(classify_npu(0x1022, 0x17F0, 0x10) == NpuKind::Strix);
    CHECK(classify_npu(0x1022, 0x17F0, 0x11) == NpuKind::Strix);
    CHECK(classify_npu(0x1022, 0x17F0, 0x20) == NpuKind::Krackan);
    CHECK(classify_npu(0x1022, 0x17F0, 0x42) == NpuKind::UnknownAmdNpu);
    CHECK(classify_npu(0x1002, 0x17F0, 0x00) == NpuKind::None);  // ATI vendor id
    CHECK(classify_npu(0x1022, 0x1234, 0x00) == NpuKind::None);
    CHECK(is_xdna2(NpuKind::Strix));
    CHECK(is_xdna2(NpuKind::Krackan));
    CHECK_FALSE(is_xdna2(NpuKind::PhoenixHawkPoint));
}

TEST_CASE("driver version comparison is numeric") {
    CHECK(compare_versions("32.0.203.280", "32.0.203.280") == 0);
    CHECK(compare_versions("32.0.203.376", "32.0.203.280") > 0);
    CHECK(compare_versions("32.0.203.99", "32.0.203.280") < 0);
    CHECK(compare_versions("32.0.203", "32.0.203.0") == 0);
    CHECK(compare_versions("31.9.999.999", "32.0.0.0") < 0);
}

TEST_CASE("backend discovery reports every backend with a reason") {
    OrtRuntime rt(ORT_LOGGING_LEVEL_ERROR);
    const auto statuses = probe_backends(rt);
    REQUIRE(statuses.size() == 5);
    bool cpu_available = false;
    for (const auto& s : statuses) {
        CHECK_FALSE(s.availability.reason.empty());
        if (s.kind == BackendKind::CPU) cpu_available = s.availability.available;
        if (s.kind == BackendKind::XDNA2 && !rt.has_provider("VitisAIExecutionProvider")) {
            CHECK_FALSE(s.availability.available);
        }
    }
    CHECK(cpu_available);
    CHECK(make_backend(BackendKind::CUDA) == nullptr);
    CHECK(make_backend(BackendKind::CPU) != nullptr);
}
