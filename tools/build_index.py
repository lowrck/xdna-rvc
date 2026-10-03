#!/usr/bin/env python3
"""Build an RVC-compatible FAISS .index from speech recordings.

Replicates upstream RVC train/train_index.py: content features of every input file
are extracted with the HuBERT content encoder, then an "IVF{n},Flat" index with
n = min(16*sqrt(N), N//39) is trained and filled (nprobe = 1). Large feature sets
(> 200k frames) are reduced to 10k k-means centers first, as upstream does.

Use this to create an index for a voice model whose .index file is missing, or to
build test fixtures.

  python tools/build_index.py --hubert assets/hubert_base --version v2 \
      --out voice.index recordings/*.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from xdna_rvc_tools import hubert
from xdna_rvc_tools.audio import load_wav_mono_16k

log = logging.getLogger("build_index")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wavs", nargs="+", type=Path)
    p.add_argument("--hubert", type=Path, required=True)
    p.add_argument("--version", choices=["v1", "v2"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    _bootstrap.setup_logging()

    import faiss

    model = hubert.build_content_encoder(args.hubert, args.version)
    feats = []
    for w in args.wavs:
        audio = load_wav_mono_16k(w)
        with torch.no_grad():
            f = model(torch.from_numpy(audio)[None])[0].numpy()
        feats.append(f)
        log.info("%s: %d frames", w.name, len(f))
    big = np.concatenate(feats).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    big = big[rng.permutation(len(big))]
    if len(big) > 200_000:
        from sklearn.cluster import MiniBatchKMeans
        big = MiniBatchKMeans(n_clusters=10000, batch_size=256 * 8, compute_labels=False,
                              init="random", random_state=args.seed).fit(big).cluster_centers_.astype(np.float32)
    n_ivf = max(1, min(int(16 * np.sqrt(len(big))), len(big) // 39))
    index = faiss.index_factory(big.shape[1], f"IVF{n_ivf},Flat")
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(big)
    for start in range(0, len(big), 8192):
        index.add(big[start:start + 8192])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out))
    log.info("wrote %s: %d vectors, dim %d, IVF%d", args.out, index.ntotal, big.shape[1], n_ivf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
