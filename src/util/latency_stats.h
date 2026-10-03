#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace xr {

struct StatsSummary {
    size_t count = 0;   // samples in the window
    double last = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double max = 0.0;
};

// Rolling window of measurements (milliseconds). push() is O(1) and does not
// allocate; summary() sorts a copy into a preallocated buffer. Not thread-safe:
// owned by one thread, summaries are published to others by value.
class RollingStats {
public:
    explicit RollingStats(size_t capacity = 512);
    void push(double value);
    StatsSummary summary() const;
    void clear();
    // Most recent values in chronological order (for graphs).
    void history(std::vector<float>& out) const;

private:
    std::vector<double> ring_;
    size_t next_ = 0;
    size_t count_ = 0;
    mutable std::vector<double> scratch_;
};

}  // namespace xr
