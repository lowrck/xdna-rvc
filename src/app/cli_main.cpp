// xdna-rvc-cli: diagnostics, offline conversion and benchmarking.

#include <cstdio>
#include <exception>

#include <CLI/CLI.hpp>

#include "app/cli_commands.h"
#include "util/error.h"

int main(int argc, char** argv) {
    CLI::App app{"xdna-rvc-cli - RVC voice conversion for AMD Ryzen AI (XDNA 2): diagnostics, offline "
                 "conversion and benchmarking"};
    app.require_subcommand(1);
    app.set_version_flag("--version", "xdna-rvc 0.1.0");

    xr::cli::GlobalOptions global;
    app.add_option("--log-level", global.log_level, "Console log level: error|warning|info|debug|trace")
        ->capture_default_str();
    app.add_option("--log-dir", global.log_dir, "Directory for log files")->capture_default_str();
    app.add_flag("--no-log-file", global.no_log_file, "Do not write a log file");
    app.add_flag("--json", global.json, "Print machine-readable JSON where supported");

    int exit_code = 0;
    xr::cli::register_providers(app, global, exit_code);
    xr::cli::register_features(app, global, exit_code);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    } catch (const xr::UserError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        if (!e.hint().empty()) std::fprintf(stderr, "hint: %s\n", e.hint().c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    return exit_code;
}
