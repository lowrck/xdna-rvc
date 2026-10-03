#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace xr::rvc {

// IVF-Flat nearest-neighbour retrieval over an exported RVC .index (see
// tools/xdna_rvc_tools/rvc/index.py for the array layout). Reproduces FAISS
// IndexIVFFlat search (squared L2, nprobe lists) and RVC's blending:
//   w = 1 / d^2 (d = squared L2 distance), normalised; retrieved = sum w * v
//   feat = retrieved * rate + feat * (1 - rate)
// CPU only. After construction, blend() does not allocate (scratch is sized in reserve()).
class FeatureIndex {
public:
    static constexpr int kTopK = 8;

    explicit FeatureIndex(const std::filesystem::path& dir);

    int dim() const { return dim_; }
    int nlist() const { return nlist_; }
    int64_t ntotal() const { return ntotal_; }
    int nprobe() const { return nprobe_; }
    void set_nprobe(int n);

    // Pre-sizes scratch for up to `max_queries` rows.
    void reserve(int max_queries);

    // Top-k search for one query. Writes up to k (distance, row) pairs sorted ascending;
    // returns the number found.
    int search(std::span<const float> query, std::span<float> dist, std::span<int64_t> rows) const;

    // In-place blend of `rows` x dim features with retrieved neighbours.
    void blend(std::span<float> feats, int rows, float rate);

    std::span<const float> vector(int64_t row) const {
        return {vectors_.data() + row * dim_, static_cast<size_t>(dim_)};
    }

private:
    int dim_ = 0;
    int nlist_ = 0;
    int64_t ntotal_ = 0;
    int nprobe_ = 1;
    std::vector<float> centroids_;
    std::vector<int64_t> offsets_;
    std::vector<float> vectors_;
    mutable std::vector<float> centroid_dist_;
    mutable std::vector<int> probe_order_;
    std::vector<float> retrieved_;
};

}  // namespace xr::rvc
