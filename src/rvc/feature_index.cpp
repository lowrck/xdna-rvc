#include "rvc/feature_index.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include "util/error.h"
#include "util/npy.h"

namespace xr::rvc {

namespace {

float l2sq(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

}  // namespace

FeatureIndex::FeatureIndex(const std::filesystem::path& dir) {
    const auto need = [&](const char* name) {
        const auto p = dir / name;
        if (!std::filesystem::exists(p)) {
            throw UserError("index file missing: " + p.string(), "Re-import the voice with --index.");
        }
        return read_npy(p);
    };
    const NpyArray c = need("centroids.npy");
    const NpyArray o = need("list_offsets.npy");
    const NpyArray v = need("vectors.npy");
    if (c.shape.size() != 2 || v.shape.size() != 2 || o.shape.size() != 1 || c.shape[1] != v.shape[1] ||
        o.shape[0] != c.shape[0] + 1) {
        throw UserError("index arrays in " + dir.string() + " have inconsistent shapes");
    }
    dim_ = static_cast<int>(v.shape[1]);
    nlist_ = static_cast<int>(c.shape[0]);
    ntotal_ = v.shape[0];
    const auto cf = c.as_float();
    centroids_.assign(cf.begin(), cf.end());
    const auto of = o.as_int64();
    offsets_.assign(of.begin(), of.end());
    const auto vf = v.as_float();
    vectors_.assign(vf.begin(), vf.end());
    if (offsets_.front() != 0 || offsets_.back() != ntotal_ || !std::is_sorted(offsets_.begin(), offsets_.end())) {
        throw UserError("index list offsets in " + dir.string() + " are corrupt");
    }
    centroid_dist_.resize(static_cast<size_t>(nlist_));
    probe_order_.resize(static_cast<size_t>(nlist_));
}

void FeatureIndex::set_nprobe(int n) { nprobe_ = std::clamp(n, 1, nlist_); }

void FeatureIndex::reserve(int max_queries) {
    retrieved_.resize(static_cast<size_t>(max_queries) * dim_);
}

int FeatureIndex::search(std::span<const float> q, std::span<float> dist, std::span<int64_t> rows) const {
    const int k = static_cast<int>(std::min(dist.size(), rows.size()));
    for (int l = 0; l < nlist_; ++l) {
        centroid_dist_[static_cast<size_t>(l)] = l2sq(q.data(), centroids_.data() + static_cast<size_t>(l) * dim_, dim_);
    }
    std::iota(probe_order_.begin(), probe_order_.end(), 0);
    const int np = std::min(nprobe_, nlist_);
    std::partial_sort(probe_order_.begin(), probe_order_.begin() + np, probe_order_.end(), [&](int a, int b) {
        const float da = centroid_dist_[static_cast<size_t>(a)], db = centroid_dist_[static_cast<size_t>(b)];
        return da < db || (da == db && a < b);
    });
    int found = 0;
    for (int i = 0; i < k; ++i) {
        dist[static_cast<size_t>(i)] = std::numeric_limits<float>::infinity();
        rows[static_cast<size_t>(i)] = -1;
    }
    for (int p = 0; p < np; ++p) {
        const int l = probe_order_[static_cast<size_t>(p)];
        for (int64_t r = offsets_[static_cast<size_t>(l)]; r < offsets_[static_cast<size_t>(l) + 1]; ++r) {
            const float d = l2sq(q.data(), vectors_.data() + r * dim_, dim_);
            if (found == k && d >= dist[static_cast<size_t>(k - 1)]) continue;
            // insertion into the sorted top-k
            int pos = std::min(found, k - 1);
            while (pos > 0 && dist[static_cast<size_t>(pos - 1)] > d) {
                dist[static_cast<size_t>(pos)] = dist[static_cast<size_t>(pos - 1)];
                rows[static_cast<size_t>(pos)] = rows[static_cast<size_t>(pos - 1)];
                --pos;
            }
            dist[static_cast<size_t>(pos)] = d;
            rows[static_cast<size_t>(pos)] = r;
            if (found < k) ++found;
        }
    }
    return found;
}

void FeatureIndex::blend(std::span<float> feats, int n, float rate) {
    if (rate <= 0.0f || n <= 0) return;
    if (retrieved_.size() < static_cast<size_t>(n) * dim_) retrieved_.resize(static_cast<size_t>(n) * dim_);
    float dist[kTopK];
    int64_t rows[kTopK];
    for (int i = 0; i < n; ++i) {
        float* f = feats.data() + static_cast<size_t>(i) * dim_;
        const int found = search(std::span<const float>(f, static_cast<size_t>(dim_)), dist, rows);
        if (found == 0) continue;
        double wsum = 0.0;
        double w[kTopK];
        for (int j = 0; j < found; ++j) {
            const double d = std::max<double>(dist[j], 1e-12);
            w[j] = 1.0 / (d * d);
            wsum += w[j];
        }
        float* out = retrieved_.data() + static_cast<size_t>(i) * dim_;
        std::fill(out, out + dim_, 0.0f);
        for (int j = 0; j < found; ++j) {
            const float wj = static_cast<float>(w[j] / wsum);
            const float* v = vectors_.data() + rows[j] * dim_;
            for (int c = 0; c < dim_; ++c) out[c] += wj * v[c];
        }
        for (int c = 0; c < dim_; ++c) f[c] = out[c] * rate + f[c] * (1.0f - rate);
    }
}

}  // namespace xr::rvc
