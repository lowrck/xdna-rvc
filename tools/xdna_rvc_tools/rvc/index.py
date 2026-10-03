"""RVC .index (FAISS) export to plain arrays, and the reference retrieval algorithm.

RVC trains "IVF{n},Flat" FAISS indexes over the training set's content features and
searches them with nprobe=1, k=8 (upstream infer/vc/pipeline.py, infer/rtrvc.py):

    score, ix = index.search(feats, k=8)          # squared L2 distances
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    retrieved = np.sum(big_npy[ix] * weight[..., None], axis=1)
    feats = retrieved * index_rate + feats * (1 - index_rate)

The native runtime implements IVF-Flat search itself, so the importer exports:
    index/centroids.npy     float32 [nlist, dim]   coarse quantizer centroids
    index/list_offsets.npy  int64   [nlist + 1]    start of each inverted list
    index/vectors.npy       float32 [ntotal, dim]  vectors grouped by list
    index/ids.npy           int64   [ntotal]       original FAISS ids (diagnostics)
    index/index.json        metadata (dim, ntotal, nlist, nprobe, source hash)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..safe_load import sha256_file


@dataclass
class IvfArrays:
    centroids: np.ndarray
    list_offsets: np.ndarray
    vectors: np.ndarray
    ids: np.ndarray
    nprobe: int

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def nlist(self) -> int:
        return int(self.centroids.shape[0])


def read_faiss_index(path: str | Path) -> IvfArrays:
    """Reads a FAISS index file (IVF-Flat or Flat) into plain arrays.

    FAISS index files are a binary format parsed by FAISS itself (no pickle); we use the
    official reader rather than re-implementing the format.
    """
    import faiss

    owner = faiss.read_index(str(path))  # keeps the C++ object alive; downcast wrappers do not own it
    index = faiss.downcast_index(owner)
    if isinstance(index, faiss.IndexFlat):
        vecs = index.reconstruct_n(0, index.ntotal).astype(np.float32)
        return IvfArrays(centroids=vecs.mean(axis=0, keepdims=True), list_offsets=np.array([0, len(vecs)], np.int64),
                         vectors=vecs, ids=np.arange(len(vecs), dtype=np.int64), nprobe=1)
    if not isinstance(index, faiss.IndexIVFFlat):
        raise ValueError(f"unsupported FAISS index type {type(index).__name__}; only IVF-Flat and Flat indexes "
                         "(as trained by RVC) are supported")
    ivf = index
    if ivf.metric_type != faiss.METRIC_L2:
        raise ValueError("only L2-metric indexes are supported")
    if ivf.ntotal == 0:
        raise ValueError("index contains no vectors; use the 'added_*.index' file, not 'trained_*.index'")
    d, nlist = ivf.d, ivf.nlist
    centroids = faiss.downcast_index(ivf.quantizer).reconstruct_n(0, nlist).astype(np.float32)
    invlists = ivf.invlists
    offsets = [0]
    vec_parts, id_parts = [], []
    for lst in range(nlist):
        size = invlists.list_size(lst)
        if size:
            ids = faiss.rev_swig_ptr(invlists.get_ids(lst), size).copy()
            codes = faiss.rev_swig_ptr(invlists.get_codes(lst), size * d * 4).copy()
            vec_parts.append(codes.view(np.float32).reshape(size, d))
            id_parts.append(ids.astype(np.int64))
        offsets.append(offsets[-1] + size)
    result = IvfArrays(centroids=centroids, list_offsets=np.asarray(offsets, np.int64),
                       vectors=np.concatenate(vec_parts).astype(np.float32), ids=np.concatenate(id_parts),
                       nprobe=max(1, int(ivf.nprobe)))
    del owner
    return result


def write_index_dir(arrays: IvfArrays, out_dir: str | Path, source: str | Path | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centroids.npy", np.ascontiguousarray(arrays.centroids, np.float32))
    np.save(out_dir / "list_offsets.npy", np.ascontiguousarray(arrays.list_offsets, np.int64))
    np.save(out_dir / "vectors.npy", np.ascontiguousarray(arrays.vectors, np.float32))
    np.save(out_dir / "ids.npy", np.ascontiguousarray(arrays.ids, np.int64))
    meta = {"dim": arrays.dim, "ntotal": int(arrays.vectors.shape[0]), "nlist": arrays.nlist,
            "nprobe": arrays.nprobe, "k": 8, "metric": "l2"}
    if source is not None:
        meta["source"] = Path(source).name
        meta["source_sha256"] = sha256_file(source)
    (out_dir / "index.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def ivf_search(arrays: IvfArrays, queries: np.ndarray, k: int = 8, nprobe: int | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Reference IVF-Flat search. Returns (squared L2 distances [n,k], row indices into
    arrays.vectors [n,k]); missing results are (inf, -1) as in FAISS."""
    nprobe = nprobe or arrays.nprobe
    q = np.asarray(queries, np.float32)
    cd = ((q[:, None, :] - arrays.centroids[None]) ** 2).sum(-1)
    probes = np.argsort(cd, axis=1, kind="stable")[:, :nprobe]
    dist = np.full((len(q), k), np.inf, np.float32)
    rows = np.full((len(q), k), -1, np.int64)
    for i in range(len(q)):
        cand = np.concatenate([np.arange(arrays.list_offsets[l], arrays.list_offsets[l + 1]) for l in probes[i]])
        if cand.size == 0:
            continue
        dd = ((arrays.vectors[cand] - q[i]) ** 2).sum(-1)
        order = np.argsort(dd, kind="stable")[:k]
        dist[i, :len(order)] = dd[order]
        rows[i, :len(order)] = cand[order]
    return dist, rows


def blend(arrays: IvfArrays, feats: np.ndarray, index_rate: float, k: int = 8, nprobe: int | None = None
          ) -> np.ndarray:
    """RVC index blending (see module docstring), guarding against zero distances."""
    if index_rate <= 0:
        return feats
    dist, rows = ivf_search(arrays, feats, k, nprobe)
    valid = rows >= 0
    w = np.where(valid, 1.0 / np.maximum(dist, 1e-12) ** 2, 0.0)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
    retrieved = (arrays.vectors[np.maximum(rows, 0)] * w[..., None]).sum(axis=1)
    return (retrieved * index_rate + feats * (1 - index_rate)).astype(np.float32)
