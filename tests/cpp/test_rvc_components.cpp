#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#include <nlohmann/json.hpp>

#include "rvc/feature_index.h"
#include "rvc/model_info.h"
#include "rvc/rmvpe.h"
#include "rvc/stream_config.h"
#include "util/error.h"
#include "util/npy.h"

using namespace xr;
using namespace xr::rvc;
namespace fs = std::filesystem;

namespace {

fs::path temp_dir(const std::string& name) {
    const auto d = fs::temp_directory_path() / "xdna_rvc_tests" / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

nlohmann::json valid_model_json() {
    return {{"schema", 1},
            {"name", "test"},
            {"conversion_version", "1.0.0"},
            {"source", {{"file", "v.pth"}, {"sha256", "abc"}}},
            {"rvc", {{"version", "v2"}, {"feature_dim", 768}, {"uses_f0", true}, {"sample_rate", 40000},
                     {"n_speakers", 2}, {"upsample_factor", 400}, {"speaker_names", {{"1", "alice"}}}}},
            {"content_encoder", {{"path", "../shared/content_encoder_v2.onnx"}}},
            {"pitch", {{"method", "rmvpe"}, {"path", "../shared/rmvpe.onnx"}}},
            {"generators",
             {{{"path", "generator_T75_s60_r15.onnx"}, {"frames", 75}, {"skip_head", 60}, {"return_length", 15},
               {"stream", {{"block_ms", 100}, {"crossfade_ms", 40}, {"extra_ms", 600}}},
               {"validation", {{"a", {{"passed", true}}}}}}}},
            {"index", {{"path", "index"}, {"dim", 768}, {"ntotal", 10}, {"nlist", 2}, {"nprobe", 1}}}};
}

}  // namespace

TEST_CASE("stream geometry matches the Python table") {
    // Same cases as tests/python/test_rvc_conversion.py::test_stream_presets_and_parsing
    auto g = make_geometry({100, 40, 600});
    CHECK(g.frames == 75);
    CHECK(g.skip_head == 60);
    CHECK(g.return_length == 15);
    CHECK(g.tag() == "T75_s60_r15");
    CHECK(make_geometry({100, 80, 500}).return_length == 15);  // SOLA buffer capped at 40 ms
    CHECK(make_geometry({80, 30, 400}).frames == 52);
    CHECK(make_geometry(preset("low_latency")).tag() == "T43_s30_r7");
    CHECK(make_geometry(preset("balanced")).tag() == "T45_s30_r9");
    CHECK(make_geometry(preset("quality")).tag() == "T75_s60_r15");
    CHECK(make_geometry({40, 20, 300, 60}).lookahead_frames() == 9);
    CHECK(g.lookahead_frames() == 5);
    CHECK_THROWS_AS(make_geometry({85, 40, 600}), std::invalid_argument);
    CHECK_THROWS_AS(make_geometry({10, 40, 600}), std::invalid_argument);
    CHECK_THROWS_AS(preset("ultra"), std::invalid_argument);
}

TEST_CASE("model.json parsing: valid file") {
    const auto dir = temp_dir("model_ok");
    std::ofstream(dir / "model.json") << valid_model_json().dump();
    const auto m = load_model_info(dir / "model.json");
    CHECK(m.name == "test");
    CHECK(m.rvc_version == "v2");
    CHECK(m.feature_dim == 768);
    CHECK(m.uses_f0);
    CHECK(m.sample_rate == 40000);
    CHECK(m.n_speakers == 2);
    CHECK(m.speaker_names.at(1) == "alice");
    CHECK(m.content_encoder == fs::absolute(dir) / "../shared/content_encoder_v2.onnx");
    CHECK(m.rmvpe_mel_basis.filename() == "rmvpe_mel_basis.npy");
    REQUIRE(m.generators.size() == 1);
    CHECK(m.generators[0].validated);
    CHECK(m.generators[0].output_samples == 15 * 400);
    CHECK(m.find_generator(make_geometry({100, 40, 600})) != nullptr);
    CHECK(m.find_generator(make_geometry({60, 30, 400})) == nullptr);
    REQUIRE(m.index.has_value());
    CHECK(m.index->dim == 768);
}

TEST_CASE("model.json parsing: actionable errors for bad files") {
    const auto dir = temp_dir("model_bad");
    auto write = [&](const nlohmann::json& j) { std::ofstream(dir / "model.json") << j.dump(); };

    CHECK_THROWS_AS(load_model_info(dir / "missing.json"), UserError);

    std::ofstream(dir / "model.json") << "{ not json";
    CHECK_THROWS_AS(load_model_info(dir / "model.json"), UserError);

    auto j = valid_model_json();
    j["schema"] = 2;
    write(j);
    CHECK_THROWS_WITH_AS(load_model_info(dir / "model.json"), doctest::Contains("schema"), UserError);

    j = valid_model_json();
    j["rvc"]["feature_dim"] = 256;  // v2 must be 768
    write(j);
    CHECK_THROWS_WITH_AS(load_model_info(dir / "model.json"), doctest::Contains("inconsistent"), UserError);

    j = valid_model_json();
    j["rvc"].erase("sample_rate");
    write(j);
    CHECK_THROWS_WITH_AS(load_model_info(dir / "model.json"), doctest::Contains("sample_rate"), UserError);

    j = valid_model_json();
    j["index"]["dim"] = 256;
    write(j);
    CHECK_THROWS_WITH_AS(load_model_info(dir / "model.json"), doctest::Contains("index dim"), UserError);

    j = valid_model_json();
    j["generators"][0]["validation"]["a"]["passed"] = false;
    write(j);
    CHECK_FALSE(load_model_info(dir / "model.json").generators[0].validated);
}

TEST_CASE("IVF index search equals brute force when all lists are probed, and blending follows RVC") {
    const auto dir = temp_dir("index");
    const int dim = 8, n = 300, nlist = 6;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    std::vector<float> vecs(static_cast<size_t>(n) * dim), cents(static_cast<size_t>(nlist) * dim);
    for (auto& v : vecs) v = nd(rng);
    for (auto& v : cents) v = nd(rng);
    // assign each vector to its nearest centroid, store grouped by list
    std::vector<std::vector<int>> lists(nlist);
    for (int i = 0; i < n; ++i) {
        int best = 0;
        float bd = 1e30f;
        for (int l = 0; l < nlist; ++l) {
            float d = 0;
            for (int c = 0; c < dim; ++c) d += std::pow(vecs[i * dim + c] - cents[l * dim + c], 2.0f);
            if (d < bd) bd = d, best = l;
        }
        lists[best].push_back(i);
    }
    std::vector<float> grouped;
    std::vector<int64_t> offsets{0};
    for (const auto& l : lists) {
        for (int i : l) grouped.insert(grouped.end(), vecs.begin() + i * dim, vecs.begin() + (i + 1) * dim);
        offsets.push_back(offsets.back() + static_cast<int64_t>(l.size()));
    }
    write_npy(dir / "centroids.npy", cents, {nlist, dim});
    write_npy(dir / "list_offsets.npy", offsets, {nlist + 1});
    write_npy(dir / "vectors.npy", grouped, {n, dim});

    FeatureIndex idx(dir);
    CHECK(idx.dim() == dim);
    CHECK(idx.ntotal() == n);
    idx.set_nprobe(nlist);  // exhaustive
    std::vector<float> q(dim);
    for (auto& v : q) v = nd(rng);
    float dist[8];
    int64_t rows[8];
    REQUIRE(idx.search(q, dist, rows) == 8);
    std::vector<float> all(n);
    for (int r = 0; r < n; ++r) {
        float d = 0;
        for (int c = 0; c < dim; ++c) d += std::pow(grouped[r * dim + c] - q[c], 2.0f);
        all[r] = d;
    }
    std::vector<float> sorted = all;
    std::sort(sorted.begin(), sorted.end());
    for (int k = 0; k < 8; ++k) {
        CHECK(dist[k] == doctest::Approx(sorted[k]));
        CHECK(all[rows[k]] == doctest::Approx(dist[k]));
    }

    // Blending: rate 0 is identity; rate 1 replaces with the 1/d^2-weighted neighbour mean.
    std::vector<float> f = q;
    idx.reserve(1);
    idx.blend(f, 1, 0.0f);
    CHECK(f == q);
    idx.blend(f, 1, 1.0f);
    double wsum = 0;
    std::vector<double> expect(dim, 0.0);
    for (int k = 0; k < 8; ++k) wsum += 1.0 / (double(dist[k]) * dist[k]);
    for (int k = 0; k < 8; ++k) {
        const double w = 1.0 / (double(dist[k]) * dist[k]) / wsum;
        for (int c = 0; c < dim; ++c) expect[c] += w * grouped[rows[k] * dim + c];
    }
    for (int c = 0; c < dim; ++c) CHECK(f[c] == doctest::Approx(expect[c]).epsilon(1e-4));
}

TEST_CASE("RMVPE decode: threshold, local average and cents mapping") {
    std::vector<float> sal(2 * RmvpePitch::kBins, 0.0f);
    // frame 0: below threshold -> unvoiced
    sal[100] = 0.02f;
    // frame 1: single peak at bin 120 -> exact bin centre
    sal[RmvpePitch::kBins + 120] = 0.9f;
    std::vector<float> f0(2);
    RmvpePitch::decode(sal, 2, f0);
    CHECK(f0[0] == 0.0f);
    const double cents = 20.0 * 120 + 1997.3794084376191;
    CHECK(f0[1] == doctest::Approx(10.0 * std::pow(2.0, cents / 1200.0)).epsilon(1e-5));
    CHECK(RmvpePitch::realtime_segment_samples(1600) == 4960);
    CHECK(RmvpePitch::frames_for_samples(4960) == 32);
    CHECK(RmvpePitch::realtime_segment_samples(6400) == 10080);
}
