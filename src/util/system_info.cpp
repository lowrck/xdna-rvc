#include "util/system_info.h"

#include <array>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#else
#include <sys/utsname.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace xr {

namespace {

#if defined(_WIN32) || defined(__x86_64__) || defined(__i386__)
void cpuid(unsigned leaf, std::array<unsigned, 4>& r) {
#if defined(_WIN32)
    int regs[4];
    __cpuid(regs, static_cast<int>(leaf));
    for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(regs[i]);
#else
    __cpuid(leaf, r[0], r[1], r[2], r[3]);
#endif
}
#endif

std::string trim(std::string s) {
    const auto b = s.find_first_not_of(" \t\0", 0, 3);
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\0", std::string::npos, 3);
    return s.substr(b, e - b + 1);
}

#if defined(_WIN32)
std::string windows_version() {
    // GetVersionEx lies without a manifest; RtlGetVersion reports the real version.
    using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll) return "Windows (unknown version)";
    auto fn = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"));
    if (!fn) return "Windows (unknown version)";
    RTL_OSVERSIONINFOW info{};
    info.dwOSVersionInfoSize = sizeof(info);
    if (fn(&info) != 0) return "Windows (unknown version)";
    const char* name = (info.dwMajorVersion == 10 && info.dwBuildNumber >= 22000) ? "Windows 11" : "Windows";
    return std::string(name) + " " + std::to_string(info.dwMajorVersion) + "." +
           std::to_string(info.dwMinorVersion) + " (build " + std::to_string(info.dwBuildNumber) + ")";
}
#endif

}  // namespace

SystemInfo query_system_info() {
    SystemInfo info;
    info.logical_cores = std::thread::hardware_concurrency();

#if defined(_WIN32) || defined(__x86_64__) || defined(__i386__)
    std::array<unsigned, 4> r{};
    cpuid(0, r);
    char vendor[13] = {};
    std::memcpy(vendor + 0, &r[1], 4);
    std::memcpy(vendor + 4, &r[3], 4);
    std::memcpy(vendor + 8, &r[2], 4);
    info.cpu_vendor = vendor;

    cpuid(0x80000000u, r);
    if (r[0] >= 0x80000004u) {
        char brand[49] = {};
        for (unsigned i = 0; i < 3; ++i) {
            cpuid(0x80000002u + i, r);
            std::memcpy(brand + i * 16, r.data(), 16);
        }
        info.cpu_brand = trim(brand);
    }
#endif
    if (info.cpu_brand.empty()) info.cpu_brand = "unknown CPU";

#if defined(_WIN32)
    info.os = windows_version();
#else
    utsname u{};
    if (uname(&u) == 0) {
        info.os = std::string(u.sysname) + " " + u.release + " " + u.machine;
    } else {
        info.os = "unknown OS";
    }
#endif
    return info;
}

}  // namespace xr
