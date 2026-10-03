// xdna-rvc-cli: diagnostics, offline conversion and benchmarking.

#include <cstdio>
#include <algorithm>
#include <exception>
#include <string>
#include <vector>

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
    app.add_option("--threads", global.threads,
                   "ONNX Runtime threads shared by all model stages (0 = automatic: min(8, cores/2))");

    int exit_code = 0;
    xr::cli::register_providers(app, global, exit_code);
    xr::cli::register_features(app, global, exit_code);
    xr::cli::register_convert(app, global, exit_code);
    xr::cli::register_inspect(app, global, exit_code);
    xr::cli::register_realtime(app, global, exit_code);

    // `xdna-rvc-cli --input in.wav --model model.json --output out.wav` is shorthand for `convert`.
    std::vector<std::string> args(argv + 1, argv + argc);
    bool has_sub = false, has_input = false;
    for (const auto& a : args) {
        if (app.get_subcommand_no_throw(a) != nullptr) has_sub = true;
        if (a == "--input" || a == "-i") has_input = true;
    }
    if (!has_sub && has_input) {
        // Global options must precede the subcommand; everything else belongs to convert.
        std::vector<std::string> reordered;
        std::vector<std::string> rest;
        for (size_t i = 0; i < args.size(); ++i) {
            const auto& a = args[i];
            if (a == "--no-log-file" || a == "--json") {
                reordered.push_back(a);
            } else if ((a == "--log-level" || a == "--log-dir" || a == "--threads") && i + 1 < args.size()) {
                reordered.push_back(a);
                reordered.push_back(args[++i]);
            } else {
                rest.push_back(a);
            }
        }
        reordered.push_back("convert");
        reordered.insert(reordered.end(), rest.begin(), rest.end());
        args = std::move(reordered);
    }
    std::reverse(args.begin(), args.end());  // CLI11 consumes a reversed vector

    try {
        app.parse(args);
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
