#include "util/latency_stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace xr {

RollingStats::RollingStats(size_t capacity) : ring_(std::max<size_t>(capacity, 1), 0.0), scratch_(ring_.size()) {}

void RollingStats::push(double value) {
    ring_[next_] = value;
    next_ = (next_ + 1) % ring_.size();
    count_ = std::min(count_ + 1, ring_.size());
}

void RollingStats::clear() {
    next_ = 0;
    count_ = 0;
}

StatsSummary RollingStats::summary() const {
    StatsSummary s;
    s.count = count_;
    if (count_ == 0) return s;
    s.last = ring_[(next_ + ring_.size() - 1) % ring_.size()];
    // Chronological order is irrelevant for order statistics.
    const size_t start = count_ < ring_.size() ? 0 : next_;
    for (size_t i = 0; i < count_; ++i) scratch_[i] = ring_[(start + i) % ring_.size()];
    auto first = scratch_.begin();
    auto last = scratch_.begin() + static_cast<std::ptrdiff_t>(count_);
    s.mean = std::accumulate(first, last, 0.0) / static_cast<double>(count_);
    std::sort(first, last);
    s.max = *(last - 1);
    s.median = count_ % 2 ? scratch_[count_ / 2] : 0.5 * (scratch_[count_ / 2 - 1] + scratch_[count_ / 2]);
    // nearest-rank p95
    const size_t rank = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(count_)));
    s.p95 = scratch_[std::clamp<size_t>(rank, 1, count_) - 1];
    return s;
}

void RollingStats::history(std::vector<float>& out) const {
    out.resize(count_);
    const size_t start = count_ < ring_.size() ? 0 : next_;
    for (size_t i = 0; i < count_; ++i) out[i] = static_cast<float>(ring_[(start + i) % ring_.size()]);
}

}  // namespace xr
