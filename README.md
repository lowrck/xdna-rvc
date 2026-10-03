# xdna-rvc

Realtime RVC (Retrieval-based Voice Conversion) voice changer for Windows, built to offload
neural inference to the **AMD XDNA 2 NPU** in Ryzen AI 300-series (Strix / Krackan) processors
through ONNX Runtime and AMD's VitisAI execution provider — with DirectML and CPU backends,
per-stage backend selection, and evidence-based reporting of what actually ran on the NPU.

Microphone → voice model → any output device (speakers, or a virtual cable into Discord, OBS,
games). Also converts WAV files offline using exactly the same streaming pipeline.

## Status

| Area | State |
|---|---|
| RVC v1/v2, F0/no-F0, 32/40/48 kHz import (.pth + .index) | Done, validated against upstream RVC |
| Offline WAV → WAV conversion | Done (CPU), measured intelligibility, pitch accuracy, alignment |
| Realtime engine (lock-free, worker thread, underrun/drift handling) | Done; verified with paced simulation on Linux |
| GUI (Dear ImGui) | Done |
| WASAPI shared/exclusive | Implemented via miniaudio; **not yet run on Windows hardware** |
| XDNA 2 (VitisAI EP, BF16) | Implemented against the Ryzen AI 1.8 API; **not yet executed on an NPU** (development machine had none) |
| DirectML | Implemented; not yet run on Windows |
| ASIO | **Not implemented** (needs the Steinberg SDK); use WASAPI exclusive |
| < 100 ms latency | **Not reached on CPU** (see [docs/latency.md](docs/latency.md)); XDNA 2 offload is the main lever |

Everything that could be tested without Windows/NPU hardware is covered by automated tests
(C++ doctest, Python pytest). Hardware-dependent tests are tagged and skip themselves with a
reason.

## Supported hardware

* **Primary:** AMD Ryzen AI 300 / Ryzen AI Max (Strix, Strix Halo) and Krackan Point — XDNA 2 NPU,
  NPU driver ≥ 32.0.203.280, Windows 11.
* **Also works:** any x64 Windows 11 PC (DirectML GPU or CPU), Linux x64 (CPU; development).
* Phoenix/Hawk Point (XDNA 1) NPUs are detected but not used (they cannot run BF16 models).

## Required software

* Windows 11, Visual Studio 2022 (Desktop C++), CMake ≥ 3.24, Git.
* Python 3.11+ for the model conversion tools.
* For the NPU: AMD **Ryzen AI Software 1.8** + NPU driver — see
  [docs/amd_xdna_setup.md](docs/amd_xdna_setup.md).
* For virtual microphone routing: a virtual audio cable (e.g. VB-Audio Virtual Cable).

## Build

```bat
:: NPU build (Ryzen AI installed, RYZEN_AI_INSTALLATION_PATH set)
cmake --preset windows-ryzenai
cmake --build --preset windows-ryzenai

:: Without the Ryzen AI SDK: DirectML + CPU
cmake --preset windows-directml
cmake --build --preset windows-directml
```

Equivalent manual form: `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` then
`cmake --build build --config Release` (the ORT flavor is picked automatically: `ryzenai` when
`RYZEN_AI_INSTALLATION_PATH` is set, otherwise `directml`). Linux: `cmake --preset linux-cpu &&
cmake --build build`. Dependencies (ONNX Runtime, spdlog, nlohmann/json, CLI11, miniaudio,
Dear ImGui, GLFW, doctest) are fetched at pinned versions; see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Outputs: `xdna-rvc` (GUI) and `xdna-rvc-cli` (diagnostics, offline conversion, benchmarks).

Run the tests: `ctest --test-dir build -C Release` and `python -m pytest`.

## Model setup

The runtime never executes PyTorch files. Voice models are converted once to ONNX by the Python
tools, which load checkpoints in **safe mode** (`weights_only`; pickled code is refused unless you
pass `--allow-unsafe-pickle` for a file you trust) and validate every export numerically.

```bat
python -m venv .venv && .venv\Scripts\activate
pip install -r tools\requirements.txt

:: 1. Shared models, once. Download from https://huggingface.co/lj1995/VoiceConversionWebUI :
::    hubert_base/ (config.json, preprocessor_config.json, pytorch_model.bin) and rmvpe.pt
python tools\convert_contentvec.py --hubert assets\hubert_base --out-dir models\shared --version v1 --version v2
python tools\convert_rmvpe.py --rmvpe assets\rmvpe.pt --out-dir models\shared

:: 2. Each voice
python tools\convert_rvc.py myvoice.pth --index added_IVF256_Flat_nprobe_1_myvoice_v2.index

:: 3. NPU only: precompile BF16 models (inside the Ryzen AI conda environment)
conda activate ryzen-ai-1.8.0
python tools\compile_xdna.py models\myvoice\model.json --preset balanced --preset low_latency
```

The GUI's **Import model** tab runs steps 1–2 for you. Details, supported checkpoint formats and
error messages: [docs/model_conversion.md](docs/model_conversion.md).

Only use voices you have the right to use, and do not impersonate real people without consent.

## Realtime use

GUI: choose Input device → Output device → Backend → RVC model → **Start**, then adjust pitch,
index rate etc. while speaking. The Realtime panel shows which backend each stage runs on,
**"XDNA 2 ACTIVE" only when NPU execution is confirmed** (VitisAI operator report or ORT profile),
measured latency, buffer levels, underruns and per-stage timings.

CLI:

```bat
xdna-rvc-cli devices
xdna-rvc-cli realtime models\myvoice\model.json --backend xdna2 --preset balanced ^
    --input-device "Microphone" --output-device "CABLE Input" --audio-api wasapi-exclusive --pitch 4
```

Stream presets (hop / crossfade / context / lookahead, chosen from measurements in
[docs/latency.md](docs/latency.md)): `low_latency` 40/20/300/60 ms, `balanced` 60/20/300/60 ms,
`quality` 100/40/600/0 ms. A preset needs a matching generator export (the importer creates
`balanced` and `low_latency`; add others with `convert_rvc.py --stream`).

## Routing to Discord, games and OBS

The app does not install a virtual microphone driver. Use an existing virtual cable:

1. Install a virtual audio cable (e.g. VB-Audio Virtual Cable). It adds a playback device
   "CABLE Input" and a recording device "CABLE Output".
2. In xdna-rvc, set **Output device = CABLE Input**.
3. In Discord / the game / OBS, set the **microphone = CABLE Output**.
4. To hear yourself, enable "Listen to this device" on CABLE Output in Windows Sound settings
   (adds latency to monitoring only).

Linux (PipeWire): `pactl load-module module-null-sink sink_name=rvc` and use "Monitor of rvc" as
the microphone in the target app.

## Offline conversion and diagnostics

```bat
xdna-rvc-cli --input test.wav --model models\myvoice\model.json --backend xdna2 --output converted.wav
xdna-rvc-cli convert test.wav out_cpu.wav --model models\myvoice\model.json --backend cpu --seed 1
xdna-rvc-cli providers                       :: NPU, driver, ORT providers, backend availability
xdna-rvc-cli inspect-model models\myvoice\model.json
xdna-rvc-cli benchmark models\myvoice\model.json --backend xdna2 --sweep --json-out bench.json
xdna-rvc-cli profile models\myvoice\model.json --backend xdna2   :: per-stage CPU/NPU placement
python tools\evaluate_conversion.py test.wav out_npu.wav --pitch 4 --compare out_cpu.wav --rmvpe assets\rmvpe.pt
```

Offline conversion streams the file through the same `StreamProcessor` used in realtime, so it
reproduces realtime behaviour exactly; `--seed` makes it deterministic for A/B comparisons
between backends.

## Documentation

* [docs/architecture.md](docs/architecture.md) — pipeline, threads, backends, design decisions
* [docs/amd_xdna_setup.md](docs/amd_xdna_setup.md) — Ryzen AI install, NPU precompilation, evidence
* [docs/model_conversion.md](docs/model_conversion.md) — import tooling and validation
* [docs/latency.md](docs/latency.md) — measured latency and quality trade-offs
* [docs/troubleshooting.md](docs/troubleshooting.md)
* [docs/implementation_plan.md](docs/implementation_plan.md) — plan, research findings, precision policy

## License

Project code: see repository license. Third-party components:
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Model weights are not distributed.
