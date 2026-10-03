#include "inference/npu_detect.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <devguid.h>
#include <setupapi.h>
#pragma comment(lib, "setupapi.lib")
#endif

namespace xr {

const char* to_string(NpuKind kind) {
    switch (kind) {
        case NpuKind::None: return "none";
        case NpuKind::PhoenixHawkPoint: return "PHX/HPT (XDNA 1)";
        case NpuKind::Strix: return "STX (XDNA 2)";
        case NpuKind::Krackan: return "KRK (XDNA 2)";
        case NpuKind::UnknownAmdNpu: return "unknown AMD NPU";
    }
    return "none";
}

bool is_xdna2(NpuKind kind) { return kind == NpuKind::Strix || kind == NpuKind::Krackan; }

const char* to_string(NpuDriverStatus status) {
    switch (status) {
        case NpuDriverStatus::NotApplicable: return "n/a";
        case NpuDriverStatus::Ok: return "ok";
        case NpuDriverStatus::Unknown: return "unknown";
        case NpuDriverStatus::TooOld: return "too old";
        case NpuDriverStatus::NotInstalled: return "not installed";
    }
    return "unknown";
}

NpuKind classify_npu(uint16_t vendor, uint16_t device, uint8_t revision) {
    if (vendor != 0x1022) return NpuKind::None;
    if (device == 0x1502) return NpuKind::PhoenixHawkPoint;
    if (device == 0x17F0) {
        if (revision == 0x20) return NpuKind::Krackan;
        if (revision == 0x00 || revision == 0x10 || revision == 0x11) return NpuKind::Strix;
        return NpuKind::UnknownAmdNpu;
    }
    return NpuKind::None;
}

int compare_versions(const std::string& a, const std::string& b) {
    auto split = [](const std::string& s) {
        std::vector<long> parts;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, '.')) parts.push_back(std::strtol(item.c_str(), nullptr, 10));
        return parts;
    };
    const auto pa = split(a);
    const auto pb = split(b);
    const size_t n = std::max(pa.size(), pb.size());
    for (size_t i = 0; i < n; ++i) {
        const long x = i < pa.size() ? pa[i] : 0;
        const long y = i < pb.size() ? pb[i] : 0;
        if (x != y) return x < y ? -1 : 1;
    }
    return 0;
}

#if defined(_WIN32)

namespace {

// Parses "PCI\VEN_1022&DEV_17F0&...&REV_10" style hardware IDs.
bool parse_hardware_id(const std::string& id, uint16_t& vendor, uint16_t& device, uint8_t& rev) {
    auto field = [&](const char* key, unsigned& out) {
        const auto pos = id.find(key);
        if (pos == std::string::npos) return false;
        out = static_cast<unsigned>(std::strtoul(id.c_str() + pos + std::strlen(key), nullptr, 16));
        return true;
    };
    unsigned v = 0, d = 0, r = 0;
    if (!field("VEN_", v) || !field("DEV_", d)) return false;
    field("REV_", r);
    vendor = static_cast<uint16_t>(v);
    device = static_cast<uint16_t>(d);
    rev = static_cast<uint8_t>(r);
    return true;
}

std::string driver_version_to_string(DWORDLONG ver) {
    return std::to_string((ver >> 48) & 0xffff) + "." + std::to_string((ver >> 32) & 0xffff) + "." +
           std::to_string((ver >> 16) & 0xffff) + "." + std::to_string(ver & 0xffff);
}

std::string registry_string(HDEVINFO set, SP_DEVINFO_DATA& data, DWORD property) {
    DWORD required = 0;
    SetupDiGetDeviceRegistryPropertyA(set, &data, property, nullptr, nullptr, 0, &required);
    if (required == 0) return {};
    std::vector<BYTE> buffer(required + 2, 0);
    if (!SetupDiGetDeviceRegistryPropertyA(set, &data, property, nullptr, buffer.data(), required, nullptr)) return {};
    return std::string(reinterpret_cast<const char*>(buffer.data()));
}

}  // namespace

// Adapted from AMD RyzenAI-SW utilities/npu_check/npu_util.cpp (MIT License,
// Copyright (c) 2023 Advanced Micro Devices, Inc.). Extended to keep the PCI
// revision so STX and KRK can be told apart.
NpuInfo detect_npu() {
    NpuInfo info;
    const GUID* classes[] = {&GUID_DEVCLASS_COMPUTEACCELERATOR, &GUID_DEVCLASS_SYSTEM};
    for (const GUID* cls : classes) {
        HDEVINFO set = SetupDiGetClassDevsA(cls, nullptr, nullptr, DIGCF_PRESENT);
        if (set == INVALID_HANDLE_VALUE) continue;
        SP_DEVINFO_DATA data{};
        data.cbSize = sizeof(data);
        for (DWORD index = 0; SetupDiEnumDeviceInfo(set, index, &data); ++index) {
            // SPDRP_HARDWAREID is a REG_MULTI_SZ; the first entry is the most specific.
            const std::string hwid = registry_string(set, data, SPDRP_HARDWAREID);
            uint16_t ven = 0, dev = 0;
            uint8_t rev = 0;
            if (!parse_hardware_id(hwid, ven, dev, rev)) continue;
            const NpuKind kind = classify_npu(ven, dev, rev);
            if (kind == NpuKind::None) continue;

            info.kind = kind;
            info.pci_vendor = ven;
            info.pci_device = dev;
            info.pci_revision = rev;
            info.device_name = registry_string(set, data, SPDRP_DEVICEDESC);

            SP_DEVINSTALL_PARAMS_A params{};
            params.cbSize = sizeof(params);
            if (SetupDiGetDeviceInstallParamsA(set, &data, &params)) {
                params.FlagsEx |= (DI_FLAGSEX_INSTALLEDDRIVER | DI_FLAGSEX_ALLOWEXCLUDEDDRVS);
                if (SetupDiSetDeviceInstallParamsA(set, &data, &params) &&
                    SetupDiBuildDriverInfoList(set, &data, SPDIT_COMPATDRIVER)) {
                    SP_DRVINFO_DATA_A drv{};
                    drv.cbSize = sizeof(drv);
                    if (SetupDiEnumDriverInfoA(set, &data, SPDIT_COMPATDRIVER, 0, &drv)) {
                        info.driver_version = driver_version_to_string(drv.DriverVersion);
                        info.driver_name = drv.Description;
                    }
                    SetupDiDestroyDriverInfoList(set, &data, SPDIT_COMPATDRIVER);
                }
            }
            break;
        }
        SetupDiDestroyDeviceInfoList(set);
        if (info.present()) break;
    }

    if (!info.present()) {
        info.detail = "No AMD NPU PCI device (VEN_1022 DEV_1502/DEV_17F0) found.";
        return info;
    }
    if (info.driver_version.empty()) {
        info.driver_status = NpuDriverStatus::NotInstalled;
        info.detail = "AMD NPU found but no installed driver version could be read. Install the NPU driver "
                      "(>= " + std::string(kMinNpuDriverVersion) + ") from AMD; see docs/amd_xdna_setup.md.";
    } else if (compare_versions(info.driver_version, kMinNpuDriverVersion) < 0) {
        info.driver_status = NpuDriverStatus::TooOld;
        info.detail = "NPU driver " + info.driver_version + " is older than the minimum " + kMinNpuDriverVersion +
                      " required by the VitisAI EP (Ryzen AI 1.5-1.8).";
    } else {
        info.driver_status = NpuDriverStatus::Ok;
        info.detail = "NPU driver " + info.driver_version + " satisfies minimum " + kMinNpuDriverVersion + ".";
    }
    return info;
}

#else  // Linux / other

namespace {

std::string read_first_line(const std::filesystem::path& p) {
    std::ifstream f(p);
    std::string line;
    if (f) std::getline(f, line);
    return line;
}

}  // namespace

NpuInfo detect_npu() {
    NpuInfo info;
    namespace fs = std::filesystem;
    const fs::path root = "/sys/bus/pci/devices";
    std::error_code ec;
    if (!fs::exists(root, ec)) {
        info.detail = "PCI sysfs not available; cannot detect an NPU on this platform.";
        return info;
    }
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        const auto ven = std::strtoul(read_first_line(entry.path() / "vendor").c_str(), nullptr, 16);
        const auto dev = std::strtoul(read_first_line(entry.path() / "device").c_str(), nullptr, 16);
        const auto rev = std::strtoul(read_first_line(entry.path() / "revision").c_str(), nullptr, 16);
        const NpuKind kind = classify_npu(static_cast<uint16_t>(ven), static_cast<uint16_t>(dev),
                                          static_cast<uint8_t>(rev));
        if (kind == NpuKind::None) continue;
        info.kind = kind;
        info.pci_vendor = static_cast<uint16_t>(ven);
        info.pci_device = static_cast<uint16_t>(dev);
        info.pci_revision = static_cast<uint8_t>(rev);
        info.device_name = entry.path().filename().string();
        const fs::path drv = entry.path() / "driver";
        if (fs::exists(drv, ec)) {
            info.driver_name = fs::read_symlink(drv, ec).filename().string();
            info.driver_status = NpuDriverStatus::Unknown;
            info.detail = "AMD NPU bound to kernel driver '" + info.driver_name +
                          "'. Note: the Ryzen AI VitisAI EP for ONNX Runtime is distributed for Windows; "
                          "this build cannot use the NPU on Linux.";
        } else {
            info.driver_status = NpuDriverStatus::NotInstalled;
            info.detail = "AMD NPU present but no kernel driver bound (amdxdna).";
        }
        return info;
    }
    info.detail = "No AMD NPU PCI device (1022:1502 or 1022:17f0) found.";
    return info;
}

#endif

}  // namespace xr
