#pragma once

#include <functional>
#include <string>
#include <vector>

namespace xr::ui {

// Runs a program with arguments, streaming combined stdout/stderr lines to `on_line`.
// Returns the exit code. Arguments are quoted; any argument containing a double quote or
// a newline is rejected (std::invalid_argument) so user-supplied paths cannot inject
// shell syntax.
int run_process(const std::vector<std::string>& argv, const std::function<void(const std::string&)>& on_line);

}  // namespace xr::ui
