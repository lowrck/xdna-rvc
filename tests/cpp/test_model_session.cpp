#include <doctest/doctest.h>

#include <array>
#include <filesystem>

#include "inference/ort_runtime.h"
#include "inference/model_session.h"
#include "util/error.h"

using namespace xr;

namespace {

const std::filesystem::path kModel = std::filesystem::path(XDNA_RVC_TEST_DATA_DIR) / "add_mul.onnx";

SessionRequest request(const std::string& stage) {
    SessionRequest r;
    r.stage = stage;
    r.model_path = kModel;
    r.diagnostics_dir = std::filesystem::temp_directory_path() / "xdna_rvc_tests" / "diag";
    return r;
}

}  // namespace

TEST_CASE("ONNX model loads on CPU and runs correctly") {
    OrtRuntime rt(ORT_LOGGING_LEVEL_ERROR);
    OpenOptions opts;
    opts.backend = BackendKind::CPU;
    opts.probe_shapes["x"] = {1, 16};
    auto s = ModelSession::open(rt, request("unit_cpu"), opts);

    REQUIRE(s->inputs().size() == 1);
    CHECK(s->inputs()[0].name == "x");
    CHECK_FALSE(s->inputs()[0].is_static());
    CHECK(s->backend() == BackendKind::CPU);
    CHECK_FALSE(s->report().fell_back);

    // Evidence probe produced a profile showing the CPU EP executed both nodes.
    REQUIRE_FALSE(s->report().provider_usage.empty());
    CHECK(s->report().provider_usage[0].provider == "CPUExecutionProvider");
    CHECK(s->report().provider_usage[0].node_count == 2);
    CHECK_FALSE(s->report().npu_evidence());

    std::array<float, 4> x{0.f, 1.f, -1.f, 2.5f};
    std::array<int64_t, 2> shape{1, 4};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<Ort::Value, 1> in{Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), shape.data(), 2)};

    // Pre-allocated output path (used by the realtime worker).
    std::array<float, 4> y{};
    std::array<Ort::Value, 1> out{Ort::Value::CreateTensor<float>(mem, y.data(), y.size(), shape.data(), 2)};
    s->run(in, out);
    CHECK(y[0] == doctest::Approx(2.f));
    CHECK(y[1] == doctest::Approx(4.f));
    CHECK(y[2] == doctest::Approx(0.f));
    CHECK(y[3] == doctest::Approx(7.f));

    // Allocating path.
    auto res = s->run(in);
    REQUIRE(res.size() == 1);
    CHECK(res[0].GetTensorData<float>()[3] == doctest::Approx(7.f));
}

TEST_CASE("missing model gives an actionable error") {
    OrtRuntime rt(ORT_LOGGING_LEVEL_ERROR);
    auto req = request("missing");
    req.model_path = "/definitely/not/here.onnx";
    CHECK_THROWS_AS(ModelSession::open(rt, req, OpenOptions{}), UserError);
}

TEST_CASE("unavailable explicit backend fails unless CPU fallback is allowed, and fallback is recorded") {
    OrtRuntime rt(ORT_LOGGING_LEVEL_ERROR);
    if (rt.has_provider("VitisAIExecutionProvider")) {
        MESSAGE("VitisAI EP present; fallback behaviour is covered by the hardware suite");
        return;
    }
    OpenOptions opts;
    opts.backend = BackendKind::XDNA2;
    opts.probe_shapes["x"] = {1, 8};
    CHECK_THROWS_AS(ModelSession::open(rt, request("strict_xdna"), opts), UserError);

    opts.allow_cpu_fallback = true;
    auto s = ModelSession::open(rt, request("fallback_xdna"), opts);
    CHECK(s->backend() == BackendKind::CPU);
    CHECK(s->report().fell_back);
    CHECK(s->report().fallback_reason.find("XDNA 2") != std::string::npos);
    CHECK(s->report().requested == BackendKind::XDNA2);
}

TEST_CASE("automatic mode picks the first available backend and does not mark it as a fallback") {
    OrtRuntime rt(ORT_LOGGING_LEVEL_ERROR);
    OpenOptions opts;
    opts.backend = BackendKind::Auto;
    opts.probe_shapes["x"] = {1, 8};
    auto s = ModelSession::open(rt, request("auto"), opts);
    CHECK(s->report().attempts.size() >= 1);
    if (!rt.has_provider("VitisAIExecutionProvider") && !rt.has_provider("DmlExecutionProvider")) {
        CHECK(s->backend() == BackendKind::CPU);
        // Unavailable backends were skipped: that is a selection, not a fallback...
        CHECK_FALSE(s->report().fell_back);
        // ...but the reason is still recorded for diagnostics.
        CHECK(s->report().attempts.front().find("unavailable") != std::string::npos);
    }
}

TEST_SUITE("hardware") {
    TEST_CASE("XDNA 2: tiny model session on the VitisAI EP") {
        OrtRuntime rt(ORT_LOGGING_LEVEL_WARNING);
        const auto avail = make_backend(BackendKind::XDNA2)->availability(rt);
        if (!avail.available) {
            MESSAGE("SKIPPED: XDNA 2 unavailable: " << avail.reason);
            return;
        }
        auto req = request("hw_xdna2");
        req.cache_dir = std::filesystem::temp_directory_path() / "xdna_rvc_tests" / "vaip_cache";
        req.cache_key = "add_mul";
        OpenOptions opts;
        opts.backend = BackendKind::XDNA2;
        opts.probe_shapes["x"] = {1, 16};
        auto s = ModelSession::open(rt, req, opts);
        CHECK(s->backend() == BackendKind::XDNA2);
        MESSAGE(s->report().summary());
    }
}
