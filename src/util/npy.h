#pragma once

// Minimal reader/writer for NumPy .npy files (format version 1.0/2.0, C order,
// little-endian float32/int64/int32). Used for exchanging tensors with the Python
// tools (validation dumps, exported index data).

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace xr {

enum class NpyDtype { Float32, Int64, Int32 };

struct NpyArray {
    NpyDtype dtype = NpyDtype::Float32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;

    size_t element_count() const;
    std::span<const float> as_float() const;
    std::span<const int64_t> as_int64() const;
    std::span<const int32_t> as_int32() const;
};

NpyArray read_npy(const std::filesystem::path& path);
void write_npy(const std::filesystem::path& path, std::span<const float> data, const std::vector<int64_t>& shape);
void write_npy(const std::filesystem::path& path, std::span<const int64_t> data, const std::vector<int64_t>& shape);

}  // namespace xr
