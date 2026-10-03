// `xdna-rvc-cli providers`: what this machine and this ONNX Runtime build can execute on.

#include <cstdio>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "app/cli_context.h"
#include "inference/inference_backend.h"

namespace xr::cli {

namespace {

int run_providers(const GlobalOptions& global) {
    CliContext ctx(global);
    OrtRuntime& rt = ctx.runtime();
    const NpuInfo& npu = rt.npu();
    const auto backends = probe_backends(rt);

    if (global.json) {
        nlohmann::json j;
        j["system"] = {{"os", ctx.system().os},
                       {"cpu", ctx.system().cpu_brand},
                       {"logical_cores", ctx.system().logical_cores}};
        j["npu"] = {{"kind", to_string(npu.kind)},
                    {"present", npu.present()},
                    {"xdna2", is_xdna2(npu.kind)},
                    {"pci", fmt::format("{:04x}:{:04x} rev {:02x}", npu.pci_vendor, npu.pci_device, npu.pci_revision)},
                    {"device_name", npu.device_name},
                    {"driver", npu.driver_name},
                    {"driver_version", npu.driver_version},
                    {"driver_status", to_string(npu.driver_status)},
                    {"detail", npu.detail}};
        j["onnxruntime"] = {{"version", rt.version()},
                            {"package_flavor", rt.build_flavor()},
                            {"available_providers", rt.available_providers()}};
        auto arr = nlohmann::json::array();
        for (const auto& b : backends) {
            arr.push_back({{"backend", std::string(to_string(b.kind))},
                           {"provider", std::string(provider_name(b.kind))},
                           {"available", b.availability.available},
                           {"reason", b.availability.reason}});
        }
        j["backends"] = arr;
        std::printf("%s\n", j.dump(2).c_str());
        return 0;
    }

    std::printf("System\n------\n");
    std::printf("  OS                : %s\n", ctx.system().os.c_str());
    std::printf("  CPU               : %s (%u logical cores)\n", ctx.system().cpu_brand.c_str(),
                ctx.system().logical_cores);
    std::printf("\nNPU\n---\n");
    std::printf("  Detected          : %s\n", to_string(npu.kind));
    if (npu.present()) {
        std::printf("  PCI ID            : %04x:%04x rev %02x\n", npu.pci_vendor, npu.pci_device, npu.pci_revision);
        if (!npu.device_name.empty()) std::printf("  Device            : %s\n", npu.device_name.c_str());
        if (!npu.driver_name.empty()) std::printf("  Driver            : %s\n", npu.driver_name.c_str());
        if (!npu.driver_version.empty()) std::printf("  Driver version    : %s\n", npu.driver_version.c_str());
        std::printf("  Driver status     : %s\n", to_string(npu.driver_status));
    }
    std::printf("  Detail            : %s\n", npu.detail.c_str());

    std::printf("\nONNX Runtime\n------------\n");
    std::printf("  Version           : %s\n", rt.version().c_str());
    std::printf("  Package flavor    : %s\n", rt.build_flavor().c_str());
    std::printf("  Providers in build:");
    for (const auto& p : rt.available_providers()) std::printf(" %s", p.c_str());
    std::printf("\n\nBackends\n--------\n");
    for (const auto& b : backends) {
        std::printf("  %-9s %-28s %-13s %s\n", std::string(display_name(b.kind)).c_str(),
                    std::string(provider_name(b.kind)).c_str(), b.availability.available ? "AVAILABLE" : "unavailable",
                    b.availability.reason.c_str());
    }
    BackendKind automatic = BackendKind::CPU;
    for (BackendKind k : automatic_preference()) {
        bool ok = false;
        for (const auto& b : backends) {
            if (b.kind == k && b.availability.available) ok = true;
        }
        if (ok) {
            automatic = k;
            break;
        }
    }
    std::printf("\n  Automatic mode would try %s first.\n", std::string(display_name(automatic)).c_str());
    std::printf("  Note: availability means the provider can be configured. Whether graph nodes actually run on\n"
                "  the NPU is verified per model (see `xdna-rvc-cli profile`).\n");
    if (!ctx.log_file().empty()) std::printf("\nLog: %s\n", ctx.log_file().string().c_str());
    return 0;
}

}  // namespace

void register_providers(CLI::App& app, GlobalOptions& global, int& exit_code) {
    auto* cmd = app.add_subcommand("providers", "Show NPU detection, ONNX Runtime providers and backend availability");
    cmd->callback([&global, &exit_code] { exit_code = run_providers(global); });
}

}  // namespace xr::cli
