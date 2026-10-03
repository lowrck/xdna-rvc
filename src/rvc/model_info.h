#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "rvc/stream_config.h"

namespace xr::rvc {

struct GeneratorVariant {
    std::filesystem::path path;  // absolute
    StreamConfig stream;
    int frames = 0;
    int skip_head = 0;
    int return_length = 0;
    int output_samples = 0;
    std::string precision;       // "fp32", later "bf16"/"int8"
    bool validated = false;      // all conversion checks passed
};

struct IndexInfo {
    std::filesystem::path dir;   // absolute
    int dim = 0;
    long long ntotal = 0;
    int nlist = 0;
    int nprobe = 1;
};

// Parsed model.json (written by tools/convert_rvc.py). All paths resolved to absolute.
struct ModelInfo {
    std::filesystem::path model_json;
    std::filesystem::path dir;
    std::string name;
    std::string conversion_version;
    std::string source_file;
    std::string source_sha256;
    std::string rvc_version;     // "v1" | "v2"
    int feature_dim = 0;
    bool uses_f0 = false;
    int sample_rate = 0;
    int n_speakers = 1;
    std::map<int, std::string> speaker_names;
    int upsample_factor = 0;     // samples per 10 ms frame at sample_rate
    int inter_channels = 192;
    std::filesystem::path content_encoder;
    std::filesystem::path rmvpe;          // empty for no-F0 models
    std::filesystem::path rmvpe_mel_basis;
    std::vector<GeneratorVariant> generators;
    std::optional<IndexInfo> index;
    std::vector<std::string> warnings;

    // Finds the generator exported for exactly this geometry, or nullptr.
    const GeneratorVariant* find_generator(const StreamGeometry& g) const;
    std::string describe() const;
};

// Throws UserError with an actionable message for malformed or incomplete model dirs.
ModelInfo load_model_info(const std::filesystem::path& model_json);

}  // namespace xr::rvc
