// xdna-rvc: desktop GUI (Dear ImGui + GLFW + OpenGL 3).

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "ui/gui_app.h"
#include "ui/log_sink.h"
#include "util/log.h"

namespace {

std::filesystem::path find_project_root(const char* argv0) {
    std::vector<std::filesystem::path> starts{std::filesystem::current_path()};
    std::error_code ec;
    const auto exe = std::filesystem::weakly_canonical(argv0, ec);
    if (!ec) starts.push_back(exe.parent_path());
    for (auto p : starts) {
        for (int i = 0; i < 6 && !p.empty(); ++i, p = p.parent_path()) {
            if (std::filesystem::exists(p / "tools" / "convert_rvc.py")) return p;
        }
    }
    return std::filesystem::current_path();
}

// Screenshot of the current back buffer as binary PPM (converted to PNG by tooling).
void save_ppm(const std::string& path, int w, int h) {
    std::vector<unsigned char> px(static_cast<size_t>(w) * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int y = h - 1; y >= 0; --y) f.write(reinterpret_cast<const char*>(px.data() + static_cast<size_t>(y) * w * 3), w * 3);
}

}  // namespace

int main(int argc, char** argv) {
    // Optional: --screenshot <file.ppm> [--tab Name] [--frames N] renders, saves and exits.
    // --simulate <wav> uses a realtime-paced file instead of audio devices; --autostart presses Start;
    // --after <s> delays the screenshot.
    std::string screenshot, tab, simulate;
    int frames_to_render = 0;
    bool autostart = false;
    double after_s = 0.0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--screenshot" && i + 1 < argc) screenshot = argv[++i];
        else if (a == "--simulate" && i + 1 < argc) simulate = argv[++i];
        else if (a == "--autostart") autostart = true;
        else if (a == "--after" && i + 1 < argc) after_s = std::stod(argv[++i]);
        else if (a == "--tab" && i + 1 < argc) tab = argv[++i];
        else if (a == "--frames" && i + 1 < argc) frames_to_render = std::stoi(argv[++i]);
    }

    const auto root = find_project_root(argv[0]);
    auto ring = std::make_shared<xr::ui::RingLogSink>();
    ring->set_pattern("%H:%M:%S [%l] %v");
    std::filesystem::create_directories(root / "logs");
    auto file = std::make_shared<spdlog::sinks::basic_file_sink_mt>((root / "logs" / "xdna-rvc-gui.log").string(), true);
    auto console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    console->set_level(spdlog::level::warn);
    auto logger = std::make_shared<spdlog::logger>("xdna-rvc", spdlog::sinks_init_list{ring, file, console});
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger);

    if (!glfwInit()) {
        std::fprintf(stderr, "error: cannot initialise GLFW (no display?)\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    if (!screenshot.empty()) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(1200, 820, "xdna-rvc - RVC voice changer for AMD Ryzen AI", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "error: cannot create an OpenGL 3.3 window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().FrameRounding = 4.0f;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    int rc = 0;
    {
        xr::ui::GuiApp::Options opts;
        opts.project_root = root;
        opts.simulate_input = simulate;
        xr::ui::GuiApp app(opts, ring);
        if (!tab.empty()) app.select_tab(tab);
        if (autostart) app.request_start();
        int frame = 0;
        const double t_start = glfwGetTime();
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            app.frame();
            ImGui::Render();
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0, 0, w, h);
            glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            ++frame;
            if (!screenshot.empty() && frame >= std::max(3, frames_to_render) && glfwGetTime() - t_start >= after_s) {
                save_ppm(screenshot, w, h);
                break;
            }
            glfwSwapBuffers(window);
            if (!screenshot.empty()) glfwWaitEventsTimeout(0.02);  // invisible window: no vsync pacing
        }
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return rc;
}
