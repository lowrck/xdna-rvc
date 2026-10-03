#pragma once

#include <string>

namespace xr {

struct SystemInfo {
    std::string os;          // e.g. "Windows 11 (build 26100)" or "Linux 6.x ..."
    std::string cpu_brand;   // CPUID brand string
    std::string cpu_vendor;  // CPUID vendor string
    unsigned logical_cores = 0;
};

SystemInfo query_system_info();

}  // namespace xr
