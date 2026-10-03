#!/usr/bin/env python3
"""Intelligibility vs streaming configuration, measured with ASR.

Converts a set of LibriSpeech utterances through the native streaming pipeline
(`xdna-rvc-cli convert`, i.e. exactly the realtime code path) for each stream
configuration and reports the corpus word error rate of wav2vec2-base-960h
transcriptions, next to the source audio's own WER. Word counts are pooled over all
utterances so a single word does not swing the result.

This measures *content preservation* (can the words still be understood), not speaker
similarity or naturalness.

  python tools/eval_stream_quality.py --model models/voice/model.json \
      --parquet test_assets/speech/librispeech_dummy.parquet --utterances 20 \
      --stream 100,40,600 --stream 40,20,300,60 --json-out quality.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from evaluate_conversion import wer
from xdna_rvc_tools.audio import load_wav, resample


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    n = len(ref.split())
    return round(wer(ref, hyp) * n), n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--parquet", type=Path, required=True, help="LibriSpeech-format parquet (audio bytes + text)")
    p.add_argument("--utterances", type=int, default=20)
    p.add_argument("--skip", type=int, default=0, help="skip the first N rows")
    p.add_argument("--stream", action="append", required=True, help="block,crossfade,extra[,lookahead] in ms")
    p.add_argument("--cli", type=Path, default=Path("build/src/xdna-rvc-cli"))
    p.add_argument("--index-rate", type=float, default=0.0)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--json-out", type=Path)
    a = p.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    rows = pq.read_table(a.parquet).to_pylist()[a.skip:a.skip + a.utterances]
    tmp = Path(tempfile.mkdtemp(prefix="xdna_rvc_eval_"))
    sources = []
    for i, r in enumerate(rows):
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        path = tmp / f"src_{i}.wav"
        sf.write(path, resample(audio, sr, 48000), 48000)
        sources.append((path, r["text"]))

    jobs = []
    for spec in a.stream:
        b, c, e, *rest = (int(x) for x in spec.split(","))
        la = rest[0] if rest else 0
        for i, (src, _) in enumerate(sources):
            out = tmp / f"out_{spec.replace(',', '_')}_{i}.wav"
            cmd = [str(a.cli), "--no-log-file", "--log-level", "error", "--threads", str(a.threads), "convert", str(src),
                   str(out), "-m", str(a.model), "--backend", "cpu", "--seed", "1", "--block-ms", str(b),
                   "--crossfade-ms", str(c), "--extra-ms", str(e), "--lookahead-ms", str(la), "--index-rate",
                   str(a.index_rate), "--output-rate", "16000"]
            jobs.append((spec, i, out, cmd))
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda j: subprocess.run(j[3], capture_output=True, text=True), jobs))
    for (spec, i, _, _), r in zip(jobs, results):
        if r.returncode != 0:
            print(f"conversion failed for {spec} #{i}: {r.stderr[-500:]}", file=sys.stderr)
            return 1

    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval()

    def transcribe(path: Path) -> str:
        x, sr = load_wav(path)
        iv = proc(resample(x, sr, 16000), sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            return proc.batch_decode(asr(iv).logits.argmax(-1))[0]

    report = {"model": str(a.model), "utterances": len(sources), "index_rate": a.index_rate, "results": []}
    errs = words = 0
    for src, text in sources:
        e, n = word_errors(text, transcribe(src))
        errs, words = errs + e, words + n
    report["source_wer"] = errs / words
    print(f"{'source audio':28s} WER {100 * errs / words:5.1f}%  ({words} words)")
    for spec in a.stream:
        errs = words = 0
        for (s, i, out, _) in jobs:
            if s != spec:
                continue
            e, n = word_errors(sources[i][1], transcribe(out))
            errs, words = errs + e, words + n
        report["results"].append({"stream": spec, "wer": errs / words, "words": words})
        print(f"stream {spec:21s} WER {100 * errs / words:5.1f}%")
    if a.json_out:
        a.json_out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
