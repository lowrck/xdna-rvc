#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <spdlog/sinks/base_sink.h>

namespace xr::ui {

// Keeps the most recent log lines for the GUI's log panel. Never used from audio callbacks
// (nothing logs there).
class RingLogSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
    explicit RingLogSink(size_t max_lines = 2000) : max_(max_lines) {}

    struct Line {
        spdlog::level::level_enum level;
        std::string text;
    };

    std::vector<Line> lines() {
        std::lock_guard lock(lines_mu_);
        return {lines_.begin(), lines_.end()};
    }
    void clear() {
        std::lock_guard lock(lines_mu_);
        lines_.clear();
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t buf;
        formatter_->format(msg, buf);
        std::string s(buf.data(), buf.size());
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        std::lock_guard lock(lines_mu_);
        lines_.push_back({msg.level, std::move(s)});
        while (lines_.size() > max_) lines_.pop_front();
    }
    void flush_() override {}

private:
    size_t max_;
    std::mutex lines_mu_;
    std::deque<Line> lines_;
};

}  // namespace xr::ui
