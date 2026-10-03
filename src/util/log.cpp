#include "util/log.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include <vector>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xr {

namespace {

spdlog::level::level_enum to_spdlog(LogLevel level) {
    switch (level) {
        case LogLevel::Trace: return spdlog::level::trace;
        case LogLevel::Debug: return spdlog::level::debug;
        case LogLevel::Info: return spdlog::level::info;
        case LogLevel::Warning: return spdlog::level::warn;
        case LogLevel::Error: return spdlog::level::err;
    }
    return spdlog::level::info;
}

std::string timestamp_for_filename() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

}  // namespace

std::string_view to_string(LogLevel level) {
    switch (level) {
        case LogLevel::Trace: return "trace";
        case LogLevel::Debug: return "debug";
        case LogLevel::Info: return "info";
        case LogLevel::Warning: return "warning";
        case LogLevel::Error: return "error";
    }
    return "info";
}

LogLevel parse_log_level(std::string_view text) {
    std::string s(text);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "trace") return LogLevel::Trace;
    if (s == "debug") return LogLevel::Debug;
    if (s == "info") return LogLevel::Info;
    if (s == "warning" || s == "warn") return LogLevel::Warning;
    if (s == "error") return LogLevel::Error;
    throw std::invalid_argument("unknown log level '" + std::string(text) +
                                "' (expected error|warning|info|debug|trace)");
}

LoggingSession::LoggingSession(const LogConfig& config) {
    std::vector<spdlog::sink_ptr> sinks;
    std::vector<std::string> deferred_warnings;  // emitted once the logger exists
    auto console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    console->set_level(to_spdlog(config.console_level));
    console->set_pattern("[%^%l%$] %v");
    sinks.push_back(console);

    if (config.log_to_file) {
        std::error_code ec;
        std::filesystem::create_directories(config.log_dir, ec);
        if (!ec) {
            log_file_ = config.log_dir / (config.file_prefix + "-" + timestamp_for_filename() + ".log");
            try {
                auto file = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_.string(), true);
                file->set_level(to_spdlog(config.file_level));
                file->set_pattern("%Y-%m-%d %H:%M:%S.%e [%l] [tid %t] %v");
                sinks.push_back(file);
            } catch (const spdlog::spdlog_ex& e) {
                log_file_.clear();
                deferred_warnings.push_back(std::string("could not open log file: ") + e.what());
            }
        } else {
            deferred_warnings.push_back("could not create log directory " + config.log_dir.string() + ": " +
                                        ec.message());
        }
    }

    logger_ = std::make_shared<spdlog::logger>("xdna-rvc", sinks.begin(), sinks.end());
    logger_->set_level(spdlog::level::trace);  // sinks filter individually
    logger_->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger_);
    spdlog::flush_every(std::chrono::seconds(2));
    for (const auto& w : deferred_warnings) logger_->warn(w);
}

LoggingSession::~LoggingSession() {
    if (logger_) logger_->flush();
    spdlog::shutdown();
}

void LoggingSession::set_console_level(LogLevel level) {
    if (logger_ && !logger_->sinks().empty()) logger_->sinks().front()->set_level(to_spdlog(level));
}

}  // namespace xr
