# AMD Ryzen AI (XDNA 2) setup

Verified against the **Ryzen AI Software 1.8.0** documentation (ryzenai.docs.amd.com, pages
"Installation Instructions", "Model Compilation and Deployment", "Application Development",
"Supported Operators", "AI Analyzer"; last updated 2026-09-28). If a newer release changes
anything below, follow AMD's documentation and update this file.

> **Status of this project's XDNA path:** implemented against the documented API, *not yet
> executed on NPU hardware* (it was developed on a machine without an NPU). The first run on a
> Strix/Krackan machine is expected to surface issues; every failure is reported with the
> stage, provider and ORT message, and CPU fallback keeps the voice changer usable.

## 1. Supported hardware

| APU | PCI ID | NPU | BF16 models | Used by this project |
|---|---|---|---|---|
| Strix / Strix Halo (Ryzen AI 300, Ryzen AI Max) | `1022:17F0` rev 00/10/11 | XDNA 2 | yes | **yes** |
| Krackan Point | `1022:17F0` rev 20 | XDNA 2 | yes | **yes** |
| Phoenix / Hawk Point (Ryzen 7040/8040) | `1022:1502` | XDNA 1 | no (INT8 only) | no |

`xdna-rvc-cli providers` prints the detected APU type, PCI ID and driver version.

## 2. Install

Requirements from AMD: Windows 11 build ≥ 22621.3527, Visual Studio 2022 with *Desktop
development with C++*, CMake ≥ 3.26, Miniforge (conda) on `PATH`.

1. **NPU driver** — version **32.0.203.280 or newer** (AMD lists 32.0.203.376 as the production
   driver for PHX/HPT/STX/KRK). Download from AMD's Ryzen AI page, unzip, and in an
   *administrator* terminal run `.\npu_sw_installer.exe`. Check: Task Manager → Performance →
   **NPU0** is listed.
2. **Ryzen AI Software** — run `ryzen-ai-1.8.0.exe` (AMD account download). Default install path
   `C:\Program Files\RyzenAI\1.8.0`; the installer sets **`RYZEN_AI_INSTALLATION_PATH`** and
   creates the conda environment **`ryzen-ai-1.8.0`**.
3. Verify:
   ```bat
   conda activate ryzen-ai-1.8.0
   cd %RYZEN_AI_INSTALLATION_PATH%\quicktest
   python quicktest.py
   ```

Driver compatibility (AMD table): VitisAI EP 1.5–1.8 need driver ≥ 32.0.203.280. The EP also
refuses drivers released after its compatibility window (EP 1.8: until 2029-07-22).
xdna-rvc implements the same checks as AMD's `npu_check` utility and reports `driver too old`
or `not installed` explicitly.

## 3. Build xdna-rvc against the Ryzen AI ONNX Runtime

Open a *new* "x64 Native Tools" prompt (so `RYZEN_AI_INSTALLATION_PATH` is visible):

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DXDNA_RVC_ORT_FLAVOR=ryzenai
cmake --build build --config Release
build\src\Release\xdna-rvc-cli.exe providers
```

`ryzenai` uses `%RYZEN_AI_INSTALLATION_PATH%\onnxruntime\include\onnxruntime\core\session`
and `...\onnxruntime\lib\onnxruntime.lib` (the layout used by AMD's own C++ samples) and copies
the deployment DLLs from `%RYZEN_AI_INSTALLATION_PATH%\deployment` next to the executables:
`onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_providers_vitisai.dll`,
`onnxruntime_vitisai_ep.dll`, `dyn_dispatch_core.dll`, `aiecompiler_client.dll`, `vaiml.dll`
(BF16), `DirectML.dll`, and others that exist in the release. CMake stops with an explicit
error if the variable or headers are missing. With `XDNA_RVC_ORT_FLAVOR=auto` (default),
Windows builds use `ryzenai` whenever `RYZEN_AI_INSTALLATION_PATH` is set.

Expected `providers` output on a working system: `VitisAIExecutionProvider` in the provider
list, `XDNA 2 ... AVAILABLE`, NPU `STX (XDNA 2)` or `KRK (XDNA 2)`, driver status `ok`.

## 4. Precompile models for the NPU (required)

AMD: *"The deployment version of the VitisAI Execution Provider does not support the
on-the-fly compilation of BF16 models. Applications utilizing BF16 models must include
pre-compiled versions."* Compilation therefore happens once per voice and stream setting, in
the Ryzen AI conda environment:

```bat
conda activate ryzen-ai-1.8.0
pip install -r tools\requirements.txt
python tools\compile_xdna.py models\<voice>\model.json --preset balanced --preset low_latency
```

For each stage (content encoder, RMVPE, generator) and stream geometry this writes a
static-shape FP32 model, compiles it to BF16 with the VitisAI EP (`config_file` = the default
`vaiml_config` from AMD's docs, `cache_dir`/`cache_key`, `enable_cache_file_io_in_mem=0`),
writes AMD's operator assignment report, compares BF16 NPU output against FP32 CPU output and
records everything under `"xdna"` in `model.json`. The runtime then loads the precompiled
cache instantly. A stage whose BF16 output exceeds tolerance is **not** placed on the NPU in
Automatic mode (it stays on DirectML/CPU); choosing `xdna2` explicitly still uses it, with a
warning.

Precision policy: **BF16 only**. INT8/A16W8 quantization is deliberately not used during
initial XDNA 2 work (see docs/implementation_plan.md).

AMD advises not to reuse caches across EP or driver versions; `model.json` records the ORT
version used, and the runtime warns on mismatch. Re-run `compile_xdna.py` after upgrading.

## 5. Which parts actually run on the NPU

Creating a VitisAI session does not mean the graph runs on the NPU. xdna-rvc collects evidence
per stage:

* **Vitis AI EP report** — the runtime sets `XLNX_ONNX_EP_REPORT_FILE=vitisai_ep_report.json`
  and `enable_cache_file_io_in_mem=0`, then parses the report in the cache directory
  (`deviceStat` node counts for `all` / `CPU` / `NPU`).
* **ORT profile** — every session is created with profiling for one probe run; the provider of
  every executed node is recorded (`VitisAIExecutionProvider` vs `CPUExecutionProvider`) with
  its share of run time.

"XDNA 2 ACTIVE" in the GUI and `[NPU nodes confirmed]` in the CLI require one of these. Tools:

```bat
xdna-rvc-cli profile models\<voice>\model.json --backend xdna2
python tools\inspect_execution.py report models\shared\xdna\cache
python tools\inspect_execution.py profile diagnostics
python tools\inspect_execution.py ops models\shared\content_encoder_v2.onnx   (static prediction, no NPU needed)
```

**AI Analyzer** (BF16 only, installed in the Ryzen AI conda env): `xdna-rvc-cli profile` sets the
provider options `ai_analyzer_visualization` and `ai_analyzer_profiling`; open the artifacts with
`aianalyzer <run directory>` to see partitions, "CPU because" reasons and the NPU timeline.

## 6. Expected operator coverage (prediction, before hardware)

From `tools/inspect_execution.py ops` against the Ryzen AI 1.8 BF16 operator table:

| Model | Nodes in BF16 table | Not in table, exact rewrite exists | No rewrite (stays on CPU) |
|---|---|---|---|
| content_encoder_v2 | 506/544 | LayerNormalization ×26, Softmax ×12 | — |
| rmvpe | 221/339 | Relu ×117 | **GRU ×1** |
| generator (40k v2) | 556/650 | LeakyRelu ×70, LayerNormalization ×12, Softmax ×6, Relu ×6 | — |

The table may understate real coverage (activations are commonly fused into neighbouring ops);
the compiler's report is authoritative. Milestone 6 adds graph rewrites for the listed ops,
each verified numerically, and evaluates splitting RMVPE so its CNN runs on the NPU and only the
GRU head runs on the CPU.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `VitisAIExecutionProvider is not in this ONNX Runtime build` | Built with the cpu/directml flavor. Rebuild with `-DXDNA_RVC_ORT_FLAVOR=ryzenai`. |
| `NPU driver ... is older than the minimum 32.0.203.280` | Install the current NPU driver (step 1). |
| `Detected PHX/HPT (XDNA 1)` | XDNA 1 cannot run BF16 models; use CPU/DirectML. |
| `prepared but not compiled` warning | Run `tools/compile_xdna.py` on the NPU machine (step 4). |
| Session creation takes minutes | First compile without cache. Precompile (step 4). |
| XDNA stage "skipped by Automatic mode: BF16 output differs from FP32" | The accuracy check failed for that stage; it runs on DirectML/CPU. Inspect with `tools/validate_onnx.py --provider VitisAIExecutionProvider`. |
| No `vitisai_ep_report.json` | Needs `enable_cache_file_io_in_mem=0` (set by xdna-rvc) and a writable cache directory. |
