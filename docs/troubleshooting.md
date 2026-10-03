# Troubleshooting

Logs: `logs/` (CLI and GUI). ORT profiles and reports: `diagnostics/`. Start with
`xdna-rvc-cli providers` and `xdna-rvc-cli inspect-model <model.json>`.

## NPU / backends

See [amd_xdna_setup.md §7](amd_xdna_setup.md#7-troubleshooting) for VitisAI-specific issues.

| Symptom | What to do |
|---|---|
| "FALLBACK: requested XDNA 2, running on CPU" | The reason follows in the message and in Diagnostics. Common: EP not in build, driver, model not precompiled. |
| "NPU active: no" although XDNA 2 selected | No node was attributed to the NPU. Run `xdna-rvc-cli profile ... --backend xdna2` and `tools/inspect_execution.py report <cache dir>`. |
| DirectML unavailable | Build with `--preset windows-directml` or `windows-ryzenai`; DirectML is Windows-only. |

## Audio

| Symptom | What to do |
|---|---|
| Underruns / crackles | GUI shows "Processing is too slow for this hop" → use `balanced` or `quality`, or a faster backend. Otherwise raise the safety buffer (Advanced). |
| "Late" counter increases | The worker stalled (system load). Use realtime priority, close heavy apps, larger hop. |
| Latency grows over time | Separate input/output devices drift; the engine corrects automatically (see drift corrections). If it keeps growing, report the log. |
| Exclusive mode fails to open | Another app holds the device or the format is unsupported; use WASAPI shared. |
| Robotic / garbled speech at small hops | Use a preset with lookahead (`balanced`, `low_latency`); see docs/latency.md. |
| Wrong voice gender/pitch | Adjust pitch shift (±12 semitones between male and female ranges). |
| Hiss or breathing in silence | Raise the noise gate (e.g. -45 dB) or lower RMS mix toward 0. |
| No sound in Discord | Output device must be the cable's *playback* side; Discord's microphone the cable's *recording* side. |

## Conversion / models

| Symptom | What to do |
|---|---|
| "no generator exported for block ..." | The chosen stream configuration needs an export: the error prints the exact `convert_rvc.py --stream` command. |
| Import fails in the GUI | The Import tab shows the Python output; check the Python path (Import tab) and `pip install -r tools/requirements.txt`. |
| Static model fails to load with "Attempt to replace the existing tensor" | Known ONNX Runtime 1.30 issue with re-optimised pinned-shape graphs; current tools no longer produce them — delete `cache/static_models` and `models/*/xdna/static` and re-run. |
