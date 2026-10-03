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
};

// Each register_* function adds a subcommand to `app`. The callback runs the command
// and sets `exit_code`.
void register_providers(CLI::App& app, GlobalOptions& global, int& exit_code);
void register_features(CLI::App& app, GlobalOptions& global, int& exit_code);

}  // namespace xr::cli
