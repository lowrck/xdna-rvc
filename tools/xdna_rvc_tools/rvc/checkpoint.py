"""Identify RVC synthesizer checkpoints from their contents, not their file names.

Supported layouts
-----------------
* inference ("small") checkpoints written by RVC's savee()/extract_small_model():
    {"weight": state_dict(fp16, no enc_q), "config": [18 hyper-parameters],
     "info": str, "sr": "40k", "f0": 0|1, "version": "v1"|"v2", ["speaker_info"]}
* training checkpoints (G_*.pth and the official pretrained/f0G40k.pth etc.):
    {"model": state_dict(incl. enc_q), "iteration": int, "learning_rate": float}
  These carry no config, so it is reconstructed from RVC's config table and verified
  against the weight shapes.
* a bare synthesizer state dict.

Architecture facts are always derived from the weights:
  feature dim      enc_p.emb_phone.weight [hidden, 256|768]   -> v1 / v2
  pitch (F0)       presence of enc_p.emb_pitch.weight
  speakers         emb_g.weight [n_speakers, gin_channels]
  vocoder          dec.ups.* kernel sizes, dec.resblocks.* layout
  attention heads  hidden / enc_p.encoder.attn_layers.0.emb_rel_k.shape[-1]
Embedded metadata is cross-checked against these and disagreements are reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..safe_load import load_checkpoint, sha256_file


class UnsupportedModelError(ValueError):
    """The checkpoint is not an RVC architecture this tool can convert."""


# RVC config table (configs/v1/*.json, configs/v2/*.json upstream). Order matches the
# "config" list in inference checkpoints:
# spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads,
# n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
# upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr
_RB = ([3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
KNOWN_CONFIGS: dict[tuple[str, int], list] = {
    ("v1", 32000): [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", *_RB, [10, 4, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 32000],
    ("v1", 40000): [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", *_RB, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
    ("v1", 48000): [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", *_RB, [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 48000],
    ("v2", 32000): [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", *_RB, [10, 8, 2, 2], 512, [20, 16, 4, 4], 109, 256, 32000],
    ("v2", 40000): [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", *_RB, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
    ("v2", 48000): [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", *_RB, [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000],
}

CONFIG_FIELDS = [
    "spec_channels", "segment_size", "inter_channels", "hidden_channels", "filter_channels", "n_heads",
    "n_layers", "kernel_size", "p_dropout", "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
    "upsample_rates", "upsample_initial_channel", "upsample_kernel_sizes", "spk_embed_dim", "gin_channels", "sr",
]

SYNTHESIZERS = {
    ("v1", True): "SynthesizerTrnMs256NSFsid",
    ("v1", False): "SynthesizerTrnMs256NSFsid_nono",
    ("v2", True): "SynthesizerTrnMs768NSFsid",
    ("v2", False): "SynthesizerTrnMs768NSFsid_nono",
}


@dataclass
class RvcCheckpointInfo:
    path: Path
    sha256: str
    layout: str                 # "inference" | "training" | "state_dict"
    version: str                # "v1" | "v2" (derived from feature dim)
    feature_dim: int            # 256 | 768
    uses_f0: bool
    sample_rate: int
    n_speakers: int
    config: dict
    synthesizer: str
    vocoder: str                # "hifigan-nsf" (f0) | "hifigan" (no f0)
    info: str = ""
    speaker_names: dict[int, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)  # raw non-weight entries (stringified)
    warnings: list[str] = field(default_factory=list)

    @property
    def upsample_factor(self) -> int:
        f = 1
        for r in self.config["upsample_rates"]:
            f *= r
        return f

    def config_list(self) -> list:
        return [self.config[k] for k in CONFIG_FIELDS]

    def to_json(self) -> dict:
        return {
            "path": str(self.path), "sha256": self.sha256, "layout": self.layout, "version": self.version,
            "feature_dim": self.feature_dim, "uses_f0": self.uses_f0, "sample_rate": self.sample_rate,
            "n_speakers": self.n_speakers, "synthesizer": self.synthesizer, "vocoder": self.vocoder,
            "upsample_factor": self.upsample_factor, "config": self.config, "info": self.info,
            "speaker_names": {str(k): v for k, v in self.speaker_names.items()}, "warnings": self.warnings,
        }


def _parse_sr(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    m = re.fullmatch(r"\s*(\d+)\s*(k?)\s*", str(value).lower())
    if not m:
        return None
    n = int(m.group(1))
    return n * 1000 if m.group(2) else n


def normalize_state_dict(sd: dict) -> dict:
    """Maps parametrization-style weight norm keys to the classic weight_g/weight_v names
    and casts everything to float32."""
    out = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        k2 = k.replace(".parametrizations.weight.original0", ".weight_g").replace(
            ".parametrizations.weight.original1", ".weight_v")
        out[k2] = v.float()
    return out


def _ups_kernels(sd: dict) -> list[int]:
    ks = []
    i = 0
    while True:
        w = sd.get(f"dec.ups.{i}.weight_v", sd.get(f"dec.ups.{i}.weight"))
        if w is None:
            break
        ks.append(int(w.shape[-1]))
        i += 1
    return ks


def _derive_from_weights(sd: dict) -> dict:
    d: dict = {}
    if "enc_p.emb_phone.weight" not in sd:
        raise UnsupportedModelError("no 'enc_p.emb_phone.weight': this is not an RVC synthesizer checkpoint")
    hidden, feat = sd["enc_p.emb_phone.weight"].shape
    d["hidden_channels"], d["feature_dim"] = int(hidden), int(feat)
    d["uses_f0"] = "enc_p.emb_pitch.weight" in sd
    if "emb_g.weight" not in sd:
        raise UnsupportedModelError("no speaker embedding 'emb_g.weight' found")
    d["n_speakers"], d["gin_channels"] = (int(x) for x in sd["emb_g.weight"].shape)
    d["inter_channels"] = int(sd["enc_p.proj.weight"].shape[0]) // 2
    n_layers = 0
    while f"enc_p.encoder.attn_layers.{n_layers}.conv_q.weight" in sd:
        n_layers += 1
    d["n_layers"] = n_layers
    d["filter_channels"] = int(sd["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[0])
    d["kernel_size"] = int(sd["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[-1])
    k_ch = int(sd["enc_p.encoder.attn_layers.0.emb_rel_k"].shape[-1])
    d["n_heads"] = d["hidden_channels"] // k_ch
    d["upsample_initial_channel"] = int(sd["dec.conv_pre.weight"].shape[0])
    d["upsample_kernel_sizes"] = _ups_kernels(sd)
    n_ups = len(d["upsample_kernel_sizes"])
    n_rb = 0
    while any(k.startswith(f"dec.resblocks.{n_rb}.") for k in sd):
        n_rb += 1
    if n_ups == 0 or n_rb % n_ups != 0:
        raise UnsupportedModelError(f"unexpected decoder layout: {n_ups} upsample layers, {n_rb} resblocks")
    per = n_rb // n_ups
    rk = []
    for j in range(per):
        w = sd.get(f"dec.resblocks.{j}.convs1.0.weight_v", sd.get(f"dec.resblocks.{j}.convs1.0.weight"))
        if w is None:
            raise UnsupportedModelError("decoder resblocks are not HiFi-GAN ResBlock1 (convs1/convs2) layout")
        rk.append(int(w.shape[-1]))
    d["resblock_kernel_sizes"] = rk
    d["resblock"] = "1" if f"dec.resblocks.0.convs2.0.weight_v" in sd or "dec.resblocks.0.convs2.0.weight" in sd else "2"
    if "enc_q.pre.weight" in sd:
        d["spec_channels"] = int(sd["enc_q.pre.weight"].shape[1])
    return d


def inspect_checkpoint(path: str | Path, sr_hint: int | None = None, allow_unsafe: bool = False
                       ) -> tuple[RvcCheckpointInfo, dict]:
    """Returns (info, float32 state dict without enc_q). Raises UnsupportedModelError."""
    path = Path(path)
    ck = load_checkpoint(path, allow_unsafe=allow_unsafe)
    if not isinstance(ck, dict):
        raise UnsupportedModelError(f"{path.name}: checkpoint is a {type(ck).__name__}, expected a dict")
    warnings: list[str] = []

    if "weight" in ck and isinstance(ck["weight"], dict):
        layout, raw_sd = "inference", ck["weight"]
    elif "model" in ck and isinstance(ck["model"], dict):
        layout, raw_sd = "training", ck["model"]
    elif any(isinstance(v, torch.Tensor) for v in ck.values()):
        layout, raw_sd = "state_dict", ck
    else:
        raise UnsupportedModelError(f"{path.name}: no 'weight' or 'model' state dict found (keys: {list(ck)[:10]})")

    metadata = {k: v for k, v in ck.items() if k not in ("weight", "model", "optimizer")}

    # Forks that changed the vocoder or embedder are recognisable by metadata.
    vocoder_meta = str(metadata.get("vocoder", "HiFi-GAN"))
    if vocoder_meta not in ("HiFi-GAN", "hifigan", "HiFiGAN"):
        raise UnsupportedModelError(
            f"{path.name}: vocoder '{vocoder_meta}' is not supported (only the standard RVC HiFi-GAN/NSF "
            "decoder is). This checkpoint comes from a fork such as Applio with a different decoder.")
    embedder = metadata.get("embedder_model")
    if embedder is not None and str(embedder).lower() not in ("contentvec", "hubert", "hubert_base"):
        warnings.append(f"checkpoint was trained with embedder '{embedder}', not ContentVec/HuBERT base; "
                        "converted output will likely sound wrong")

    sd = normalize_state_dict(raw_sd)
    sd = {k: v for k, v in sd.items() if not k.startswith("enc_q.")} | {
        k: v for k, v in sd.items() if k == "enc_q.pre.weight"}  # keep only for spec_channels detection
    d = _derive_from_weights(sd)
    sd.pop("enc_q.pre.weight", None)

    if d["feature_dim"] == 256:
        version = "v1"
    elif d["feature_dim"] == 768:
        version = "v2"
    else:
        raise UnsupportedModelError(f"{path.name}: content feature dim {d['feature_dim']} is neither 256 (v1) nor 768 (v2)")
    meta_version = metadata.get("version")
    if meta_version is not None and meta_version != version:
        warnings.append(f"metadata says version '{meta_version}' but weights have {d['feature_dim']}-dim features "
                        f"({version}); using {version}")
    meta_f0 = metadata.get("f0")
    if meta_f0 is not None and bool(int(meta_f0)) != d["uses_f0"]:
        warnings.append(f"metadata f0={meta_f0} disagrees with weights (pitch embedding "
                        f"{'present' if d['uses_f0'] else 'absent'}); trusting weights")

    # Configuration: embedded list (inference layout) or reconstructed (training layout).
    config: dict | None = None
    if isinstance(metadata.get("config"), (list, tuple)) and len(metadata["config"]) == len(CONFIG_FIELDS):
        config = dict(zip(CONFIG_FIELDS, list(metadata["config"])))
        config["sr"] = _parse_sr(config["sr"])
    sr = _parse_sr(metadata.get("sr")) or (config["sr"] if config else None) or sr_hint
    if sr is None:
        m = re.search(r"(32|40|48)k", path.name, re.IGNORECASE)
        if m:
            sr = int(m.group(1)) * 1000
            warnings.append(f"sample rate {sr} taken from file name; pass --sample-rate to override")
    if config is None:
        candidates = [(key, cfg) for key, cfg in KNOWN_CONFIGS.items() if key[0] == version and (sr is None or key[1] == sr)]
        matches = []
        for key, cfg in candidates:
            c = dict(zip(CONFIG_FIELDS, cfg))
            if (c["upsample_kernel_sizes"] == d["upsample_kernel_sizes"]
                    and ("spec_channels" not in d or c["spec_channels"] == d["spec_channels"])):
                matches.append((key, c))
        if len(matches) != 1:
            raise UnsupportedModelError(
                f"{path.name}: cannot determine the model configuration (version {version}, sample rate "
                f"{sr or 'unknown'}, upsample kernels {d['upsample_kernel_sizes']}). "
                + ("Pass --sample-rate 32000|40000|48000." if len(matches) > 1 or sr is None else
                   "No known RVC configuration matches these weights."))
        (_, sr), config = matches[0]
        layout_note = "reconstructed from the RVC config table"
        warnings.append(f"no embedded config ({layout} checkpoint); {layout_note} for {version}/{sr}")
    if sr is None:
        sr = config["sr"]
    config["sr"] = int(sr)

    # Cross-check config against the weights (the weights win).
    config["spk_embed_dim"] = d["n_speakers"]  # RVC itself overrides this the same way
    for key in ("inter_channels", "hidden_channels", "filter_channels", "n_heads", "n_layers", "kernel_size",
                "upsample_initial_channel", "upsample_kernel_sizes", "resblock_kernel_sizes", "gin_channels", "resblock"):
        if key in d and config.get(key) != d[key]:
            raise UnsupportedModelError(f"{path.name}: config {key}={config.get(key)} does not match weights ({d[key]}); "
                                        "the checkpoint uses an unknown RVC variant")
    up = 1
    for r in config["upsample_rates"]:
        up *= r
    if up * 100 != config["sr"]:
        raise UnsupportedModelError(f"{path.name}: upsample factor {up} does not give 100 frames/s at {config['sr']} Hz")

    speaker_names = {}
    for item in metadata.get("speaker_info", []) or []:
        try:
            speaker_names[int(item["id"])] = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue

    info = RvcCheckpointInfo(
        path=path, sha256=sha256_file(path), layout=layout, version=version, feature_dim=d["feature_dim"],
        uses_f0=d["uses_f0"], sample_rate=config["sr"], n_speakers=d["n_speakers"], config=config,
        synthesizer=SYNTHESIZERS[(version, d["uses_f0"])], vocoder="hifigan-nsf" if d["uses_f0"] else "hifigan",
        info=str(metadata.get("info", "")), speaker_names=speaker_names,
        metadata={k: (v if isinstance(v, (int, float, str)) else repr(v)[:200]) for k, v in metadata.items()
                  if k not in ("config", "speaker_info")},
        warnings=warnings,
    )
    return info, sd


def build_upstream_synthesizer(info: RvcCheckpointInfo, sd: dict) -> torch.nn.Module:
    """Instantiates the upstream RVC synthesizer (reference implementation) with weights loaded."""
    from .upstream import models as upstream_models

    cls = getattr(upstream_models, info.synthesizer)
    net = cls(*info.config_list(), is_half=False)
    del net.enc_q
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise UnsupportedModelError(
            f"{info.path.name}: state dict does not fit {info.synthesizer}: missing {missing[:6]}, unexpected {unexpected[:6]}")
    return net.float().eval()


def make_inference_checkpoint(info: RvcCheckpointInfo, sd: dict, out: str | Path) -> Path:
    """Writes an RVC 'small' inference checkpoint (as RVC's extract_small_model does): fp16
    weights without enc_q, plus config/sr/f0/version metadata."""
    out = Path(out)
    torch.save({
        "weight": {k: v.half() for k, v in sd.items()},
        "config": info.config_list(),
        "info": info.info or "extracted by xdna-rvc",
        "sr": f"{info.sample_rate // 1000}k",
        "f0": int(info.uses_f0),
        "version": info.version,
    }, out)
    return out
