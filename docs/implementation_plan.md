# Implementation Plan

This is the working plan for xdna-rvc. It is kept short on purpose; design detail
lives in `architecture.md`, and setup detail in `amd_xdna_setup.md`.

## Development environment reality

The first development machine is **Linux (CachyOS), AMD Ryzen 9 9950X3D, no NPU**.
Consequences:

- Everything is built and tested on Linux with GCC/Clang and the CPU execution
  provider. The code is portable C++20; Windows-only pieces (WASAPI, SetupAPI NPU
  detection, DirectML, Vitis AI EP DLL deployment, ASIO) are compiled only on
  Windows and are kept small and isolated.
- The XDNA 2 path is implemented against the documented Ryzen AI 1.8 API
  (`AppendExecutionProvider_VitisAI`, provider options, EP report file,
  EP context cache) but **cannot be executed here**. Hardware-dependent tests are
  marked and skipped, never deleted.

## Research findings that change the original brief

| Brief assumption | Current reality (verified Oct 2026) | Action |
|---|---|---|
| "Vitis AI EP" | Ryzen AI Software **1.8.0**, provider `VitisAIExecutionProvider` | Use it; detect at runtime |
| FP32 models just run on NPU | FP32 CNN/Transformer models are converted to **BF16** by the VAIML compiler on STX/KRK. The *deployment* EP **cannot compile BF16 on the fly**; BF16 models must be precompiled (Python, Ryzen AI conda env) and loaded from the VitisAI EP cache. | `tools/compile_xdna.py` precompiles; the app consumes the cache and reports clearly when it is missing |
| `log_level` provider option | Deprecated; use ORT session `log_severity_level` | Use session option |
| Operator support | No **GRU** in the support table; Relu/LeakyRelu/Softmax/LayerNormalization/Gelu absent from the BF16 column | RMVPE BiGRU stays on CPU; graph rewrites (LeakyRelu→Max/Mul etc.) are a Milestone 6 experiment with numerical verification |
| Which nodes run on NPU | `XLNX_ONNX_EP_REPORT_FILE` + `enable_cache_file_io_in_mem=0` produces a JSON report with CPU/NPU node counts; ORT profiling records the provider of every executed node | Both are parsed and shown; "XDNA 2 ACTIVE" requires evidence |
| Device detection | PCI `1022:1502` = PHX/HPT, `1022:17F0` = STX/KRK (rev 0x20 = KRK). Min NPU driver 32.0.203.280 for EP 1.5–1.8 | Implemented in `npu_detect` |
| RVC uses fairseq HuBERT | Upstream RVC now loads a **Transformers** `HubertModel` (`hubert_base/` with `pytorch_model.bin`) | Content-encoder conversion accepts the Transformers checkpoint, loaded with `weights_only=True` |
| RVC SineGen cumsum at audio rate | Upstream now integrates phase at frame rate | Use upstream formulation; noise injected as explicit model inputs so validation is deterministic |
| Upstream realtime design | RVC's realtime GUI runs inference *inside the audio callback* | Not copied: we use ring buffers + a worker thread |

## Milestones (status tracked in README)

1. Repo, CMake, logging, CLI, ORT wrapper, provider enumeration, CPU backend → `xdna-rvc-cli providers`.
2. ContentVec/HuBERT → ONNX; C++ offline feature extraction; Python vs C++ comparison.
3. RVC `.pth` → `generator.onnx` with checkpoint inspection, PyTorch vs ORT validation.
4. Full offline WAV → WAV (RMVPE, index, generator) on CPU through the *streaming* processor used by realtime.
5. XDNA 2 backend integration, per-stage tests, assignment evidence.
6. Graph optimisation for XDNA 2 with numerical verification after each rewrite.
7. Realtime audio engine (miniaudio: WASAPI shared/exclusive, ALSA/Pulse for dev), worker thread, counters.
8. XDNA 2 in realtime, latency profiling.
9. ImGui GUI.
10. Latency optimisation.

## Key design decisions

- **One streaming processor** (`rvc::StreamProcessor`) is used by realtime audio, offline WAV conversion and
  benchmarking, so offline output is produced by exactly the realtime neural path.
- **Fixed shapes** per streaming configuration. Converted generator variants are keyed by
  (window frames, skip head, return length). A dynamic-shape generator is kept for analysis.
- **Per-stage backends**: content encoder, pitch, generator each own an ORT session with an independently
  selected backend. `Automatic` tries XDNA 2 → DirectML → CPU per stage and logs every decision.
- **Index retrieval**: the Python importer exports the FAISS IVF index into `.npy` arrays (centroids,
  inverted lists, vectors); C++ implements IVF-Flat search (nprobe configurable, default 1 like RVC).
  This keeps FAISS/BLAS out of the native build. Python tests compare against FAISS.

## Precision policy (user directive, 2026-10-02)

- **BF16 is the XDNA 2 target.** Models are exported FP32; the Ryzen AI VAIML compiler converts them to
  BF16 when precompiling for STX/KRK.
- **No INT8 / A16W8 quantization** of the content encoder, RMVPE or synthesizer during initial XDNA 2
  work. INT8 may be introduced later *per stage*, only after that stage is validated in BF16, and is
  rejected if it causes meaningful audible degradation.
- **FP32 CPU is the reference.** Every precision/backend change is regression-tested against it
  numerically (stage outputs) and on audio (tools/evaluate_conversion.py: pitch tracking, alignment,
  spectral distance, ASR WER).
- **Mixed placement is allowed.** A BF16-sensitive stage or partition may run FP16 on DirectML or FP32 on
  CPU while the remaining stages stay BF16 on XDNA 2 (per-stage backends already exist).
