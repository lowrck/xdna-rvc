#include "rvc/model_info.h"

#include <fstream>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "util/error.h"

namespace xr::rvc {

namespace {

using nlohmann::json;

template <typename T>
T required(const json& j, const char* key, const std::filesystem::path& file) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        throw UserError(fmt::format("{}: missing required field '{}'", file.string(), key),
                        "Re-import the voice with tools/convert_rvc.py.");
    }
    try {
        return it->get<T>();
    } catch (const json::exception& e) {
        throw UserError(fmt::format("{}: field '{}' has the wrong type: {}", file.string(), key, e.what()));
    }
}

}  // namespace

const GeneratorVariant* ModelInfo::find_generator(const StreamGeometry& g) const {
    for (const auto& v : generators) {
        if (v.frames == g.frames && v.skip_head == g.skip_head && v.return_length == g.return_length) return &v;
    }
    return nullptr;
}

const XdnaEntry* ModelInfo::find_xdna(const std::string& stage, const std::filesystem::path& source,
                                      const std::map<std::string, int64_t>& dims) const {
    std::error_code ec;
    for (const auto& e : xdna) {
        if (e.stage == stage && e.dims == dims && std::filesystem::equivalent(e.source, source, ec)) return &e;
    }
    return nullptr;
}

std::string ModelInfo::describe() const {
    std::string s = fmt::format("{}: RVC {} ({}-dim), {}, {} Hz, {} speaker(s), {} generator variant(s), index: {}",
                                name, rvc_version, feature_dim, uses_f0 ? "F0" : "no F0", sample_rate, n_speakers,
                                generators.size(),
                                index ? fmt::format("{} vectors, IVF{}", index->ntotal, index->nlist) : "none");
    return s;
}

ModelInfo load_model_info(const std::filesystem::path& model_json) {
    std::ifstream in(model_json);
    if (!in) {
        throw UserError("cannot open model description " + model_json.string(),
                        "Pass the model.json written by tools/convert_rvc.py (models/<voice>/model.json).");
    }
    json j;
    try {
        in >> j;
    } catch (const json::exception& e) {
        throw UserError(fmt::format("{} is not valid JSON: {}", model_json.string(), e.what()));
    }
    ModelInfo m;
    m.model_json = std::filesystem::absolute(model_json);
    m.dir = m.model_json.parent_path();
    const int schema = required<int>(j, "schema", model_json);
    if (schema != 1) {
        throw UserError(fmt::format("{}: unsupported model.json schema {} (this build reads schema 1)",
                                    model_json.string(), schema));
    }
    m.name = required<std::string>(j, "name", model_json);
    m.conversion_version = j.value("conversion_version", "");
    if (const auto src = j.find("source"); src != j.end() && src->is_object()) {
        m.source_file = src->value("file", "");
        m.source_sha256 = src->value("sha256", "");
    }
    const json rvc = required<json>(j, "rvc", model_json);
    m.rvc_version = required<std::string>(rvc, "version", model_json);
    m.feature_dim = required<int>(rvc, "feature_dim", model_json);
    m.uses_f0 = required<bool>(rvc, "uses_f0", model_json);
    m.sample_rate = required<int>(rvc, "sample_rate", model_json);
    m.n_speakers = rvc.value("n_speakers", 1);
    m.upsample_factor = required<int>(rvc, "upsample_factor", model_json);
    m.inter_channels = rvc.value("inter_channels", 192);
    if (const auto names = rvc.find("speaker_names"); names != rvc.end() && names->is_object()) {
        for (const auto& [k, v] : names->items()) m.speaker_names[std::stoi(k)] = v.get<std::string>();
    }
    if (m.rvc_version != "v1" && m.rvc_version != "v2") {
        throw UserError(fmt::format("{}: unknown RVC version '{}'", model_json.string(), m.rvc_version));
    }
    if ((m.rvc_version == "v1" && m.feature_dim != 256) || (m.rvc_version == "v2" && m.feature_dim != 768)) {
        throw UserError(fmt::format("{}: RVC {} with feature_dim {} is inconsistent", model_json.string(),
                                    m.rvc_version, m.feature_dim));
    }
    if (m.upsample_factor * 100 != m.sample_rate) {
        throw UserError(fmt::format("{}: upsample_factor {} does not match sample_rate {}", model_json.string(),
                                    m.upsample_factor, m.sample_rate));
    }

    m.content_encoder = m.dir / required<std::string>(required<json>(j, "content_encoder", model_json), "path",
                                                      model_json);
    if (m.uses_f0) {
        const json pitch = required<json>(j, "pitch", model_json);
        const std::string method = pitch.value("method", "rmvpe");
        if (method != "rmvpe") {
            throw UserError(fmt::format("{}: pitch method '{}' is not supported (only rmvpe)", model_json.string(),
                                        method));
        }
        m.rmvpe = m.dir / required<std::string>(pitch, "path", model_json);
        m.rmvpe_mel_basis = m.rmvpe.parent_path() / "rmvpe_mel_basis.npy";
    }
    for (const auto& g : required<json>(j, "generators", model_json)) {
        GeneratorVariant v;
        v.path = m.dir / required<std::string>(g, "path", model_json);
        v.frames = required<int>(g, "frames", model_json);
        v.skip_head = required<int>(g, "skip_head", model_json);
        v.return_length = required<int>(g, "return_length", model_json);
        v.output_samples = g.value("output_samples", v.return_length * m.upsample_factor);
        v.precision = g.value("precision", "fp32");
        if (const auto s = g.find("stream"); s != g.end()) {
            v.stream = {s->value("block_ms", 0), s->value("crossfade_ms", 0), s->value("extra_ms", 0)};
        }
        v.validated = true;
        if (const auto val = g.find("validation"); val != g.end() && val->is_object()) {
            for (const auto& [name, r] : val->items()) {
                if (!r.value("passed", false)) v.validated = false;
            }
        }
        m.generators.push_back(std::move(v));
    }
    if (m.generators.empty()) {
        throw UserError(model_json.string() + " lists no generator exports", "Re-run tools/convert_rvc.py.");
    }
    if (const auto idx = j.find("index"); idx != j.end() && idx->is_object()) {
        IndexInfo ii;
        ii.dir = m.dir / idx->value("path", "index");
        ii.dim = idx->value("dim", 0);
        ii.ntotal = idx->value("ntotal", 0LL);
        ii.nlist = idx->value("nlist", 0);
        ii.nprobe = idx->value("nprobe", 1);
        if (ii.dim != m.feature_dim) {
            throw UserError(fmt::format("{}: index dim {} does not match feature dim {}", model_json.string(), ii.dim,
                                        m.feature_dim));
        }
        m.index = ii;
    }
    if (const auto x = j.find("xdna"); x != j.end() && x->is_object()) {
        for (const auto& e : x->value("entries", json::array())) {
            XdnaEntry xe;
            xe.stage = e.value("stage", "");
            const json dims = e.value("dims", json::object());
            for (const auto& [k, v] : dims.items()) xe.dims[k] = v.get<int64_t>();
            xe.source = std::filesystem::weakly_canonical(m.dir / e.value("source", ""));
            xe.static_model = m.dir / e.value("static_model", "");
            xe.cache_dir = m.dir / e.value("cache_dir", "");
            xe.cache_key = e.value("cache_key", "");
            xe.config_file = m.dir / e.value("config_file", "");
            xe.precision = e.value("precision", "bf16");
            xe.onnxruntime_version = e.value("onnxruntime_version", "");
            xe.compiled = e.value("compiled", false);
            if (const auto acc = e.find("bf16_vs_fp32"); acc != e.end() && acc->is_object()) {
                xe.accuracy_passed = acc->value("passed", false);
                xe.max_rel_rms = acc->value("max_rel_rms", 0.0);
            }
            m.xdna.push_back(std::move(xe));
        }
    }
    if (const auto w = j.find("warnings"); w != j.end() && w->is_array()) {
        for (const auto& s : *w) m.warnings.push_back(s.get<std::string>());
    }
    return m;
}

}  // namespace xr::rvc
