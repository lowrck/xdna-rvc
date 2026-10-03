# Latency: model, measurements and trade-offs

All numbers below were **measured** on the development machine (AMD Ryzen 9 9950X3D, Linux,
ONNX Runtime 1.30 CPU, 8 shared ORT threads, test voice `vctk_v2_f0_40k`) on 2026-10-02.
No NPU was available, so these are CPU-only results; they are a baseline for comparing XDNA 2
runs, not a prediction of them. Reproduce with the commands at the end.

## Where the latency comes from

For every hop the realtime engine measures how long the first sample of the hop waits
between the microphone and the speaker:

```
latency = hop                       (waiting for a full hop of input)
        + processing time           (neural + DSP for this hop)
        + output queue level        (≈ safety buffer once steady)
        + algorithmic delay         (crossfade + SOLA search + lookahead + resampler filters)
        + device buffers            (input + output, reported by the driver)
```

In steady state *processing + queue* ≈ the output pre-fill, i.e. `hop + safety`, so

```
latency ≈ hop + safety + algorithmic + devices
```

* **safety** must exceed the processing time's tail or playback underruns. With `--safety-ms -1`
  (default) it is set to 1.25 × the slowest warm-up hop + 10 ms. Underruns that still happen
  add their missing samples to the queue; the engine trims that excess back out during silence.
* **algorithmic** = crossfade + 10 ms SOLA search − 5 ms (mean SOLA offset) + lookahead + ≈2 ms
  of resampler filters.
* **devices**: WASAPI shared typically 10 + 10 ms; WASAPI exclusive 3 + 3 ms is achievable.

## Intelligibility vs stream configuration

`tools/eval_stream_quality.py`: 20 held-out LibriSpeech utterances (230 words), converted by
the native pipeline, transcribed by wav2vec2-base-960h; index retrieval off. Source audio WER
4.3%. (This measures content preservation, not voice similarity.)

| hop | crossfade | context | lookahead | WER |
|---:|---:|---:|---:|---:|
| 100 | 40 | 600 | 0 | 18.7% |
| 100 | 40 | 300 | 0 | 20.9% |
| 60 | 30 | 400 | 0 | 27.4% |
| 60 | 20 | 300 | 0 | 30.4% |
| 60 | 20 | 300 | 30 | 29.6% |
| **60** | **20** | **300** | **60** | **18.7%** |
| 40 | 20 | 300 | 0 | 36.1% |
| 40 | 20 | 300 | 30 | 33.9% |
| **40** | **20** | **300** | **60** | **20.0%** |
| 40 | 20 | 300 | 100 | 21.7% |

Finding: small hops lose intelligibility mainly because the decoded frames sit at the very end
of the window, where HuBERT (a bidirectional transformer) and the synthesizer's attention see
no future audio. Upstream RVC has the same structure. 60 ms of lookahead restores the quality of
the 100 ms / 600 ms configuration; 30 ms does not. The presets follow from this:

| preset | hop | crossfade | context | lookahead | algorithmic |
|---|---:|---:|---:|---:|---:|
| `low_latency` | 40 | 20 | 300 | 60 | 86.6 ms |
| `balanced` | 60 | 20 | 300 | 60 | 86.6 ms |
| `quality` | 100 | 40 | 600 | 0 | 46.6 ms |

## Processing time per hop (CPU only)

`xdna-rvc-cli benchmark --sweep` (median ms; RMVPE runs concurrently with the content encoder):

| preset | content encoder | generator | total median | total p95 | hop budget |
|---|---:|---:|---:|---:|---:|
| low_latency | ~15 | ~12 | 33.9 | 36.3 | 40 |
| balanced | ~15 | ~13 | 35.4 | 45.8 | 60 |
| quality | ~16 | ~21 | 48.6 | 52.6 | 100 |

Content encoder and generator costs are dominated by fixed per-run overhead rather than window
length (300 ms vs 600 ms context changes little), and ORT scaling saturates at ~8 threads on
this CPU (16 threads is slower). `low_latency` is **CPU-bound** here: p95 is within 10% of the
40 ms hop and occasional underruns occur. It is intended for XDNA 2.

## Measured end-to-end latency (CPU, simulated 10 + 10 ms devices)

`xdna-rvc-cli realtime --simulate <speech.wav>` (realtime-paced, full model):

| preset | measured latency (median) | underruns |
|---|---:|---:|
| low_latency | 204–246 ms | 0–4 per 10 s (CPU-bound) |
| balanced | 228–235 ms | 0 |
| quality | 248 ms | 0 |
| (60/20/300, no lookahead: worse quality) | 154 ms | 0 |
| (40/20/300, no lookahead: worse quality) | 142 ms | 0 |

## Status against the < 100 ms goal

Not reached on CPU, and the measurements show why: at quality-preserving settings the
algorithmic delay alone is ~87 ms (of which 60 ms is lookahead), before adding the hop, the
safety buffer and device buffers. Paths forward, each to be measured rather than assumed:

1. **XDNA 2 / DirectML offload** shrinks processing time, which shrinks the safety buffer and
   makes the 40 ms hop sustainable. Content encoder on the NPU also runs concurrently with
   RMVPE on the CPU (already implemented).
2. **WASAPI exclusive** with 3 ms periods saves ~14 ms over shared mode.
3. **Less lookahead with a better-trained model**: the lookahead requirement measured here is
   for the VCTK base generator; fine-tuned voices may need less. Re-run `eval_stream_quality.py`
   per voice.
4. A content encoder with less dependence on future frames (e.g. a causal/streaming
   distillation) would remove most of the lookahead; out of scope for this project version.

## Reproduce

```bash
python tools/convert_rvc.py voice.pth --index voice.index --stream 40,20,300,60 --stream 60,20,300,0
xdna-rvc-cli benchmark models/voice/model.json --sweep --input speech.wav --json-out bench.json
xdna-rvc-cli realtime models/voice/model.json --preset balanced --simulate speech.wav
python tools/eval_stream_quality.py --model models/voice/model.json \
    --parquet librispeech.parquet --utterances 20 --stream 40,20,300,60 --stream 100,40,600
```
