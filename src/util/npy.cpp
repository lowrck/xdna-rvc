#include "util/npy.h"

#include <cstring>
#include <fstream>
#include <numeric>
#include <regex>
#include <stdexcept>

namespace xr {

namespace {

size_t dtype_size(NpyDtype d) {
    switch (d) {
        case NpyDtype::Float32: return 4;
        case NpyDtype::Int64: return 8;
        case NpyDtype::Int32: return 4;
    }
    return 4;
}

const char* dtype_descr(NpyDtype d) {
    switch (d) {
        case NpyDtype::Float32: return "<f4";
        case NpyDtype::Int64: return "<i8";
        case NpyDtype::Int32: return "<i4";
    }
    return "<f4";
}

void write_impl(const std::filesystem::path& path, const void* data, size_t bytes, NpyDtype dtype,
                const std::vector<int64_t>& shape) {
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        shape_str += (shape.size() == 1 || i + 1 < shape.size()) ? "," : "";
        if (i + 1 < shape.size()) shape_str += " ";
    }
    shape_str += ")";
    std::string header = std::string("{'descr': '") + dtype_descr(dtype) + "', 'fortran_order': False, 'shape': " +
                         shape_str + ", }";
    // magic(6) + version(2) + header_len(2) + header + '\n' must be a multiple of 64.
    const size_t base = 10;
    size_t total = base + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    if (header.size() > 65535) throw std::runtime_error("npy header too long");

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path.string());
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
    f.write(magic, 8);
    const uint16_t hlen = static_cast<uint16_t>(header.size());
    const char hl[2] = {static_cast<char>(hlen & 0xff), static_cast<char>(hlen >> 8)};
    f.write(hl, 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("write failed: " + path.string());
}

}  // namespace

size_t NpyArray::element_count() const {
    return std::accumulate(shape.begin(), shape.end(), size_t{1},
                           [](size_t a, int64_t b) { return a * static_cast<size_t>(b); });
}

std::span<const float> NpyArray::as_float() const {
    if (dtype != NpyDtype::Float32) throw std::runtime_error("npy array is not float32");
    return {reinterpret_cast<const float*>(bytes.data()), element_count()};
}

std::span<const int64_t> NpyArray::as_int64() const {
    if (dtype != NpyDtype::Int64) throw std::runtime_error("npy array is not int64");
    return {reinterpret_cast<const int64_t*>(bytes.data()), element_count()};
}

std::span<const int32_t> NpyArray::as_int32() const {
    if (dtype != NpyDtype::Int32) throw std::runtime_error("npy array is not int32");
    return {reinterpret_cast<const int32_t*>(bytes.data()), element_count()};
}

NpyArray read_npy(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path.string() + ": not a .npy file");
    const int major = magic[6];
    uint32_t hlen = 0;
    if (major == 1) {
        unsigned char b[2];
        f.read(reinterpret_cast<char*>(b), 2);
        hlen = b[0] | (b[1] << 8);
    } else if (major == 2 || major == 3) {
        unsigned char b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
    } else {
        throw std::runtime_error(path.string() + ": unsupported .npy version");
    }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    if (!f) throw std::runtime_error(path.string() + ": truncated header");

    NpyArray arr;
    std::smatch m;
    if (!std::regex_search(header, m, std::regex(R"('descr':\s*'([^']+)')")))
        throw std::runtime_error(path.string() + ": missing descr");
    const std::string descr = m[1];
    if (descr == "<f4") arr.dtype = NpyDtype::Float32;
    else if (descr == "<i8") arr.dtype = NpyDtype::Int64;
    else if (descr == "<i4") arr.dtype = NpyDtype::Int32;
    else throw std::runtime_error(path.string() + ": unsupported dtype " + descr);
    if (std::regex_search(header, m, std::regex(R"('fortran_order':\s*True)")))
        throw std::runtime_error(path.string() + ": Fortran-ordered arrays are not supported");
    if (!std::regex_search(header, m, std::regex(R"('shape':\s*\(([^)]*)\))")))
        throw std::runtime_error(path.string() + ": missing shape");
    const std::string dims = m[1];
    std::regex num(R"(\d+)");
    for (auto it = std::sregex_iterator(dims.begin(), dims.end(), num); it != std::sregex_iterator(); ++it)
        arr.shape.push_back(std::stoll(it->str()));

    const size_t bytes = arr.element_count() * dtype_size(arr.dtype);
    arr.bytes.resize(bytes);
    f.read(reinterpret_cast<char*>(arr.bytes.data()), static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(f.gcount()) != bytes) throw std::runtime_error(path.string() + ": truncated data");
    return arr;
}

void write_npy(const std::filesystem::path& path, std::span<const float> data, const std::vector<int64_t>& shape) {
    write_impl(path, data.data(), data.size_bytes(), NpyDtype::Float32, shape);
}

void write_npy(const std::filesystem::path& path, std::span<const int64_t> data, const std::vector<int64_t>& shape) {
    write_impl(path, data.data(), data.size_bytes(), NpyDtype::Int64, shape);
}

}  // namespace xr
