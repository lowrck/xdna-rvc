#pragma once

#include <cstdint>
#include <string>

namespace xr {

// AMD NPU (XDNA) detection, following the Ryzen AI 1.8 "Application Development"
// requirements: identify the APU type from the PCI hardware ID and verify the NPU
// driver version before using the VitisAI execution provider.
//
//   PCI 1022:1502 rev 00        -> PHX or HPT  (XDNA 1, "AIE2")
//   PCI 1022:17F0 rev 00/10/11  -> STX         (XDNA 2, "AIE2P")
//   PCI 1022:17F0 rev 20        -> KRK         (XDNA 2)
enum class NpuKind { None, PhoenixHawkPoint, Strix, Krackan, UnknownAmdNpu };

const char* to_string(NpuKind kind);
bool is_xdna2(NpuKind kind);

enum class NpuDriverStatus {
    NotApplicable,   // no NPU present
    Ok,
    Unknown,         // could not read the driver version
    TooOld,          // below the minimum for the VitisAI EP version we target
    NotInstalled,    // device present but no driver bound
};

const char* to_string(NpuDriverStatus status);

struct NpuInfo {
    NpuKind kind = NpuKind::None;
    uint16_t pci_vendor = 0;
    uint16_t pci_device = 0;
    uint8_t pci_revision = 0;
    std::string device_name;     // OS device description, if available
    std::string driver_name;     // e.g. "amdxdna" on Linux
    std::string driver_version;  // e.g. "32.0.203.280" on Windows
    NpuDriverStatus driver_status = NpuDriverStatus::NotApplicable;
    std::string detail;          // human readable explanation of the result

    bool present() const { return kind != NpuKind::None; }
};

// Minimum NPU driver for VitisAI EP 1.5 .. 1.8 (Ryzen AI docs, "VitisAI EP / NPU Driver Compatibility").
inline constexpr const char* kMinNpuDriverVersion = "32.0.203.280";

NpuInfo detect_npu();

// Exposed for tests.
NpuKind classify_npu(uint16_t vendor, uint16_t device, uint8_t revision);
// Compares dotted versions numerically ("32.0.203.280"). Returns <0, 0, >0.
int compare_versions(const std::string& a, const std::string& b);

}  // namespace xr
