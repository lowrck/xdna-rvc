#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>

namespace xr::audio {

// Single-producer / single-consumer lock-free ring of floats.
// Wait-free for both sides; no allocation after construction. One thread may call
// write()/write_available(), one other thread read()/read_available()/discard().
class SpscRing {
public:
    explicit SpscRing(size_t min_capacity) {
        size_t cap = 1;
        while (cap < min_capacity + 1) cap <<= 1;
        capacity_ = cap;
        mask_ = cap - 1;
        buf_ = std::make_unique<float[]>(cap);
        std::fill(buf_.get(), buf_.get() + cap, 0.0f);
    }

    size_t capacity() const { return capacity_ - 1; }

    size_t read_available() const {
        const size_t w = write_.load(std::memory_order_acquire);
        const size_t r = read_.load(std::memory_order_relaxed);
        return (w - r) & mask_;
    }
    size_t write_available() const {
        const size_t w = write_.load(std::memory_order_relaxed);
        const size_t r = read_.load(std::memory_order_acquire);
        return capacity_ - 1 - ((w - r) & mask_);
    }

    // Writes up to data.size() samples; returns the number written.
    size_t write(std::span<const float> data) {
        const size_t w = write_.load(std::memory_order_relaxed);
        const size_t r = read_.load(std::memory_order_acquire);
        const size_t space = capacity_ - 1 - ((w - r) & mask_);
        const size_t n = std::min(space, data.size());
        const size_t first = std::min(n, capacity_ - (w & mask_));
        std::memcpy(buf_.get() + (w & mask_), data.data(), first * sizeof(float));
        std::memcpy(buf_.get(), data.data() + first, (n - first) * sizeof(float));
        write_.store(w + n, std::memory_order_release);
        return n;
    }

    // Reads up to out.size() samples; returns the number read.
    size_t read(std::span<float> out) {
        const size_t r = read_.load(std::memory_order_relaxed);
        const size_t w = write_.load(std::memory_order_acquire);
        const size_t avail = (w - r) & mask_;
        const size_t n = std::min(avail, out.size());
        const size_t first = std::min(n, capacity_ - (r & mask_));
        std::memcpy(out.data(), buf_.get() + (r & mask_), first * sizeof(float));
        std::memcpy(out.data() + first, buf_.get(), (n - first) * sizeof(float));
        read_.store(r + n, std::memory_order_release);
        return n;
    }

    // Drops up to n samples from the read side (consumer only).
    size_t discard(size_t n) {
        const size_t r = read_.load(std::memory_order_relaxed);
        const size_t w = write_.load(std::memory_order_acquire);
        const size_t k = std::min(n, (w - r) & mask_);
        read_.store(r + k, std::memory_order_release);
        return k;
    }

private:
    size_t capacity_ = 0;
    size_t mask_ = 0;
    std::unique_ptr<float[]> buf_;
    // Monotonic counters (wrap naturally); separate cache lines to avoid false sharing.
    alignas(64) std::atomic<size_t> write_{0};
    alignas(64) std::atomic<size_t> read_{0};
};

}  // namespace xr::audio
