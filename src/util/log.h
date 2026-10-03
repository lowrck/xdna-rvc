#pragma once

// Structured logging built on spdlog.
//
// RULE: nothing in this header may be used from the realtime audio callback.
// The audio thread communicates through counters and lock-free queues only.

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include <spdlog/spdlog.h>

namespace xr {

enum class LogLevel { Trace, Debug, Info, Warning, Error };

std::string_view to_string(LogLevel level);
// Accepts error|warning|warn|info|debug|trace (case-insensitive). Throws std::invalid_argument.
LogLevel parse_log_level(std::string_view text);

struct LogConfig {
    LogLevel console_level = LogLevel::Info;
    LogLevel file_level = LogLevel::Debug;
    std::filesystem::path log_dir = "logs";
    std::string file_prefix = "xdna-rvc";
    bool log_to_file = true;
};

// Owns the process logger. Construct once in main(); destruction flushes.
class LoggingSession {
public:
    explicit LoggingSession(const LogConfig& config);
    ~LoggingSession();
    LoggingSession(const LoggingSession&) = delete;
    LoggingSession& operator=(const LoggingSession&) = delete;

    const std::filesystem::path& log_file() const { return log_file_; }
    void set_console_level(LogLevel level);

private:
    std::shared_ptr<spdlog::logger> logger_;
    std::filesystem::path log_file_;
};

}  // namespace xr

#define XR_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define XR_LOG_DEBUG(...) ::spdlog::debug(__VA_ARGS__)
#define XR_LOG_INFO(...) ::spdlog::info(__VA_ARGS__)
#define XR_LOG_WARN(...) ::spdlog::warn(__VA_ARGS__)
#define XR_LOG_ERROR(...) ::spdlog::error(__VA_ARGS__)
