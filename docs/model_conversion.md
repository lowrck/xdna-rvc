# Model conversion

All tools run with Python 3.11+ (`pip install -r tools/requirements.txt`). Checkpoints are loaded
with `torch.load(weights_only=True)`: a file that needs arbitrary pickled objects is refused with
`UnsafeCheckpointError` unless `--allow-unsafe-pickle` is given (only for files you trust; it
executes code from the file). The test suite includes a malicious-pickle test.

## Shared models (once)

| Tool | Input | Output |
|---|---|---|
| `convert_contentvec.py` | `hubert_base/` (Transformers: config.json + pytorch_model.bin or model.safetensors) or fairseq `hubert_base.pt` (needs `--allow-unsafe-pickle`) | `content_encoder_v1.onnx` (layer 9 + final_proj, 256-d), `content_encoder_v2.onnx` (layer 12, 768-d) |
| `convert_rmvpe.py` | `rmvpe.pt` | `rmvpe.onnx` (mel → salience), `rmvpe_mel_basis.npy` |

Both write `*.validation.json` with max/mean absolute and relative errors on noise and (with
`--validate-wav`) speech, and exit non-zero on failure. `--reference-transformers` also checks the
content encoder against `transformers.HubertModel`, which is what upstream RVC uses.

## Voices

```bash
python tools/convert_rvc.py voice.pth [--index voice.index] [--name NAME] \
    [--preset balanced --preset low_latency --preset quality] [--stream 40,20,300,60] \
    [--sample-rate 40000] [--inspect-only] [--force]
```

What it does:

1. **Inspect** — architecture is derived from the weights: feature dim (256 → v1, 768 → v2), pitch
   embedding (F0 or not), speaker count, decoder layout, attention heads. Embedded metadata
   (`version`, `f0`, `sr`, `config`) is cross-checked; disagreements become warnings (weights win).
   Supported layouts: RVC inference checkpoints (`weight`/`config`/`sr`/`f0`/`version`), training
   checkpoints (`G_*.pth`, official pretrained; config reconstructed from RVC's config table and
   verified against weight shapes; sample rate from metadata, `--sample-rate` or the file name),
   bare state dicts. Unsupported: non-HiFi-GAN vocoders from forks (e.g. RefineGAN) — clear error.
2. **Rebuild** the upstream RVC synthesizer (vendored, MIT) and load weights strictly.
3. **Export** per stream geometry a fixed-shape model (`GeneratorExport`): mask-free, static
   relative attention, explicit `rnd`/`noise` inputs, weight norm folded.
4. **Validate** export model vs upstream `infer()` (same noise) and ONNX Runtime vs PyTorch.
5. **Index** — FAISS IVF-Flat lists exported to `index/*.npy`; the native search algorithm is
   checked against FAISS (neighbour agreement ≥ 99%).
6. **Write** `model.json` and `conversion_report.json`. Re-running with an unchanged checkpoint
   reuses existing exports; XDNA precompile entries are preserved.

`model.json` fields: name, conversion_version, source (file, sha256, layout), rvc (version,
feature_dim, uses_f0, sample_rate, n_speakers, speaker_names, upsample_factor), content_encoder,
pitch, generators[] (path, stream, frames, skip_head, return_length, inputs, precision, sha256,
validation), index (dim, ntotal, nlist, nprobe), xdna (precompiled entries), warnings.

Missing `.index`? `tools/build_index.py` rebuilds one from recordings of the target voice using
upstream RVC's recipe (IVF n = min(16·√N, N/39), nprobe 1).

## Errors you may see

| Message | Meaning |
|---|---|
| `cannot be loaded with weights_only=True (safe mode)` | File contains pickled objects; see above |
| `vocoder 'RefineGAN' is not supported` | Fork with a different decoder |
| `cannot determine the model configuration ... Pass --sample-rate` | Training checkpoint without metadata |
| `config X does not match weights` | Unknown RVC variant (non-standard hyper-parameters) |
| `index dim 768 does not match the model's 256-dim features` | v1 voice with a v2 index (or vice versa) |
| `VALIDATION FAILED` | Export does not reproduce PyTorch; do not use it, report the conversion_report.json |

## Test voices used during development

No voice models are distributed. Development used RVC's official pretrained generators
(VCTK-based, `pretrained/` and `pretrained_v2/` from lj1995/VoiceConversionWebUI) converted to the
inference layout with `make_inference_checkpoint`, plus indexes built with `build_index.py` from
LibriSpeech (CC BY 4.0) test audio.
