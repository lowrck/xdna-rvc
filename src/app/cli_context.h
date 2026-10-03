#pragma once

#include <memory>

#include "app/cli_commands.h"
#include "inference/ort_runtime.h"
#include "util/log.h"
#include "util/system_info.h"

namespace xr::cli {

// Logging + ONNX Runtime + system facts, created by each subcommand that needs them.
class CliContext {
public:
    explicit CliContext(const GlobalOptions& global);

    OrtRuntime& runtime() { return *runtime_; }
    const SystemInfo& system() const { return system_; }
    const std::filesystem::path& log_file() const { return logging_->log_file(); }

private:
    std::unique_ptr<LoggingSession> logging_;
    SystemInfo system_;
    std::unique_ptr<OrtRuntime> runtime_;
};

}  // namespace xr::cli
