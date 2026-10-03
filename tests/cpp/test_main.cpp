#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    // Library code logs through spdlog; keep test output readable.
    spdlog::set_level(spdlog::level::warn);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    return ctx.run();
}
