#include "app/cli_context.h"

#include <algorithm>

namespace xr::cli {

CliContext::CliContext(const GlobalOptions& global) {
    LogConfig cfg;
    cfg.console_level = parse_log_level(global.log_level);
    cfg.log_dir = global.log_dir;
    cfg.log_to_file = !global.no_log_file;
    cfg.file_prefix = "xdna-rvc-cli";
    logging_ = std::make_unique<LoggingSession>(cfg);

    system_ = query_system_info();
    XR_LOG_DEBUG("xdna-rvc-cli 0.1.0");
    XR_LOG_INFO("OS: {}", system_.os);
    XR_LOG_INFO("CPU: {} ({} logical cores)", system_.cpu_brand, system_.logical_cores);
    int threads = global.threads;
    if (threads <= 0) threads = std::clamp(static_cast<int>(system_.logical_cores) / 4, 1, 8);  // ~physical/2
    runtime_ = std::make_unique<OrtRuntime>(ORT_LOGGING_LEVEL_WARNING, threads);
}

}  // namespace xr::cli
