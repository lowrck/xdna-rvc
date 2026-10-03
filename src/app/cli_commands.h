#pragma once

#include <string>

#include <CLI/CLI.hpp>

namespace xr::cli {

// Shared options available to every subcommand.
struct GlobalOptions {
    std::string log_level = "info";
    std::string log_dir = "logs";
    bool no_log_file = false;
    bool json = false;  // machine readable output where supported
    int threads = 0;    // ONNX Runtime global intra-op pool size; 0 = automatic
};

// Each register_* function adds a subcommand to `app`. The callback runs the command
// and sets `exit_code`.
void register_providers(CLI::App& app, GlobalOptions& global, int& exit_code);
void register_features(CLI::App& app, GlobalOptions& global, int& exit_code);
void register_convert(CLI::App& app, GlobalOptions& global, int& exit_code);
void register_inspect(CLI::App& app, GlobalOptions& global, int& exit_code);

}  // namespace xr::cli
