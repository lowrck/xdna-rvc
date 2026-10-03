#include "ui/process.h"

#include <cstdio>
#include <stdexcept>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace xr::ui {

int run_process(const std::vector<std::string>& argv, const std::function<void(const std::string&)>& on_line) {
    std::string cmd;
    for (const auto& a : argv) {
        if (a.find('"') != std::string::npos || a.find('\n') != std::string::npos) {
            throw std::invalid_argument("argument contains a quote or newline: " + a);
        }
#if !defined(_WIN32)
        if (a.find('$') != std::string::npos || a.find('`') != std::string::npos) {
            throw std::invalid_argument("argument contains shell metacharacters: " + a);
        }
#endif
        cmd += (cmd.empty() ? "\"" : " \"") + a + "\"";
    }
    cmd += " 2>&1";
#if defined(_WIN32)
    // cmd.exe strips the outer quotes of a command line that starts with a quote.
    cmd = "\"" + cmd + "\"";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) throw std::runtime_error("cannot start: " + cmd);
    char buf[4096];
    std::string line;
    while (fgets(buf, sizeof(buf), pipe)) {
        line += buf;
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
            on_line(line);
            line.clear();
        }
    }
    if (!line.empty()) on_line(line);
#if defined(_WIN32)
    return _pclose(pipe);
#else
    const int status = pclose(pipe);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

}  // namespace xr::ui
