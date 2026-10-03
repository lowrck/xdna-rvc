"""Self-contained HuBERT-base (ContentVec) content encoder for RVC.

Why not export transformers.HubertModel directly? Owning the module lets us control
the exported graph: no attention-mask plumbing, no dropout/SpecAugment branches,
weight norm folded into a plain Conv, fixed output layer, and simple ops that are
easier to partition onto the XDNA NPU. Numerical equivalence to the reference
implementation is checked by `compare_with_transformers()` and by the test suite.

RVC semantics (upstream infer/hubert.py):
  v1: final_proj(hidden_states[9])  -> 256-dim features
  v2: last_hidden_state (12 layers) -> 768-dim features
Frame rate: 50 Hz at 16 kHz input (320 samples per frame, receptive field 400).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .safe_load import load_checkpoint

CONV_DIM = [512] * 7
CONV_KERNEL = [10, 3, 3, 3, 3, 2, 2]
CONV_STRIDE = [5, 2, 2, 2, 2, 2, 2]
HIDDEN = 768
HEADS = 12
FFN = 3072
LAYERS = 12
POS_KERNEL = 128
POS_GROUPS = 16
EPS = 1e-5
SAMPLE_RATE = 16000
HOP = 320


def num_frames(num_samples: int) -> int:
    """Number of output frames for an input of num_samples (conv stack arithmetic)."""
    n = num_samples
    for k, s in zip(CONV_KERNEL, CONV_STRIDE):
        n = (n - k) // s + 1
    return n


class EncoderLayer(nn.Module):
    """Post-LN transformer layer (do_stable_layer_norm=False)."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN)
        self.attn_norm = nn.LayerNorm(HIDDEN, eps=EPS)
        self.fc1 = nn.Linear(HIDDEN, FFN)
        self.fc2 = nn.Linear(FFN, HIDDEN)
        self.final_norm = nn.LayerNorm(HIDDEN, eps=EPS)
        self.head_dim = HIDDEN // HEADS
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, C]
        b, t, _ = x.shape
        q = (self.q_proj(x) * self.scale).view(b, t, HEADS, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, HEADS, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, HEADS, self.head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, HIDDEN)
        x = self.attn_norm(x + self.out_proj(y))
        x = self.final_norm(x + self.fc2(F.gelu(self.fc1(x))))
        return x


class HubertContentEncoder(nn.Module):
    """Raw 16 kHz audio [B, N] -> content features [B, T, D]."""

    def __init__(self, output_layer: int = 12, final_proj: bool = False) -> None:
        super().__init__()
        if not 1 <= output_layer <= LAYERS:
            raise ValueError(f"output_layer must be in 1..{LAYERS}")
        self.output_layer = output_layer
        self.use_final_proj = final_proj
        convs = []
        in_ch = 1
        for i, (dim, k, s) in enumerate(zip(CONV_DIM, CONV_KERNEL, CONV_STRIDE)):
            convs.append(nn.Conv1d(in_ch, dim, k, s, bias=False))
            in_ch = dim
        self.convs = nn.ModuleList(convs)
        self.conv0_norm = nn.GroupNorm(CONV_DIM[0], CONV_DIM[0], eps=EPS, affine=True)
        self.feat_norm = nn.LayerNorm(CONV_DIM[-1], eps=EPS)
        self.feat_proj = nn.Linear(CONV_DIM[-1], HIDDEN)
        # weight norm is folded into a plain conv at load time
        self.pos_conv = nn.Conv1d(HIDDEN, HIDDEN, POS_KERNEL, padding=POS_KERNEL // 2, groups=POS_GROUPS)
        self.enc_norm = nn.LayerNorm(HIDDEN, eps=EPS)
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(output_layer))
        self.final_proj = nn.Linear(HIDDEN, 256) if final_proj else None

    @property
    def feature_dim(self) -> int:
        return 256 if self.use_final_proj else HIDDEN

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        x = source.unsqueeze(1)  # [B, 1, N]
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i == 0:
                x = self.conv0_norm(x)
            x = F.gelu(x)
        x = x.transpose(1, 2)  # [B, T, 512]
        x = self.feat_proj(self.feat_norm(x))
        pos = self.pos_conv(x.transpose(1, 2))
        pos = F.gelu(pos[:, :, :-1])  # SamePad: drop the extra frame of the even-length kernel
        x = self.enc_norm(x + pos.transpose(1, 2))
        for layer in self.layers:
            x = layer(x)
        if self.final_proj is not None:
            x = self.final_proj(x)
        return x


def _fold_weight_norm(g: torch.Tensor, v: torch.Tensor, dim: int) -> torch.Tensor:
    dims = [d for d in range(v.dim()) if d != dim]
    return g * v / torch.linalg.vector_norm(v, dim=dims, keepdim=True)


def _map_transformers(sd: dict) -> dict:
    out = {}
    for i in range(7):
        out[f"convs.{i}.weight"] = sd[f"feature_extractor.conv_layers.{i}.conv.weight"]
    out["conv0_norm.weight"] = sd["feature_extractor.conv_layers.0.layer_norm.weight"]
    out["conv0_norm.bias"] = sd["feature_extractor.conv_layers.0.layer_norm.bias"]
    out["feat_norm.weight"] = sd["feature_projection.layer_norm.weight"]
    out["feat_norm.bias"] = sd["feature_projection.layer_norm.bias"]
    out["feat_proj.weight"] = sd["feature_projection.projection.weight"]
    out["feat_proj.bias"] = sd["feature_projection.projection.bias"]
    p = "encoder.pos_conv_embed.conv."
    if p + "parametrizations.weight.original0" in sd:
        g, v = sd[p + "parametrizations.weight.original0"], sd[p + "parametrizations.weight.original1"]
    else:
        g, v = sd[p + "weight_g"], sd[p + "weight_v"]
    out["pos_conv.weight"] = _fold_weight_norm(g.float(), v.float(), dim=2)
    out["pos_conv.bias"] = sd[p + "bias"]
    out["enc_norm.weight"] = sd["encoder.layer_norm.weight"]
    out["enc_norm.bias"] = sd["encoder.layer_norm.bias"]
    for i in range(LAYERS):
        s, d = f"encoder.layers.{i}.", f"layers.{i}."
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            out[d + n + ".weight"] = sd[s + f"attention.{n}.weight"]
            out[d + n + ".bias"] = sd[s + f"attention.{n}.bias"]
        out[d + "attn_norm.weight"] = sd[s + "layer_norm.weight"]
        out[d + "attn_norm.bias"] = sd[s + "layer_norm.bias"]
        out[d + "fc1.weight"] = sd[s + "feed_forward.intermediate_dense.weight"]
        out[d + "fc1.bias"] = sd[s + "feed_forward.intermediate_dense.bias"]
        out[d + "fc2.weight"] = sd[s + "feed_forward.output_dense.weight"]
        out[d + "fc2.bias"] = sd[s + "feed_forward.output_dense.bias"]
        out[d + "final_norm.weight"] = sd[s + "final_layer_norm.weight"]
        out[d + "final_norm.bias"] = sd[s + "final_layer_norm.bias"]
    if "final_proj.weight" in sd:
        out["final_proj.weight"] = sd["final_proj.weight"]
        out["final_proj.bias"] = sd["final_proj.bias"]
    return out


def _map_fairseq(sd: dict) -> dict:
    out = {}
    for i in range(7):
        out[f"convs.{i}.weight"] = sd[f"feature_extractor.conv_layers.{i}.0.weight"]
    out["conv0_norm.weight"] = sd["feature_extractor.conv_layers.0.2.weight"]
    out["conv0_norm.bias"] = sd["feature_extractor.conv_layers.0.2.bias"]
    out["feat_norm.weight"] = sd["layer_norm.weight"]
    out["feat_norm.bias"] = sd["layer_norm.bias"]
    out["feat_proj.weight"] = sd["post_extract_proj.weight"]
    out["feat_proj.bias"] = sd["post_extract_proj.bias"]
    out["pos_conv.weight"] = _fold_weight_norm(sd["encoder.pos_conv.0.weight_g"].float(),
                                               sd["encoder.pos_conv.0.weight_v"].float(), dim=2)
    out["pos_conv.bias"] = sd["encoder.pos_conv.0.bias"]
    out["enc_norm.weight"] = sd["encoder.layer_norm.weight"]
    out["enc_norm.bias"] = sd["encoder.layer_norm.bias"]
    for i in range(LAYERS):
        s, d = f"encoder.layers.{i}.", f"layers.{i}."
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            out[d + n + ".weight"] = sd[s + f"self_attn.{n}.weight"]
            out[d + n + ".bias"] = sd[s + f"self_attn.{n}.bias"]
        out[d + "attn_norm.weight"] = sd[s + "self_attn_layer_norm.weight"]
        out[d + "attn_norm.bias"] = sd[s + "self_attn_layer_norm.bias"]
        for n in ("fc1", "fc2"):
            out[d + n + ".weight"] = sd[s + f"{n}.weight"]
            out[d + n + ".bias"] = sd[s + f"{n}.bias"]
        out[d + "final_norm.weight"] = sd[s + "final_layer_norm.weight"]
        out[d + "final_norm.bias"] = sd[s + "final_layer_norm.bias"]
    if "final_proj.weight" in sd:
        out["final_proj.weight"] = sd["final_proj.weight"]
        out["final_proj.bias"] = sd["final_proj.bias"]
    return out


def load_state_dict_any(path: str | Path, allow_unsafe: bool = False) -> tuple[dict, str]:
    """Returns (state_dict, format) where format is 'transformers' or 'fairseq'.

    `path` may be a Transformers directory (containing pytorch_model.bin or
    model.safetensors) or a single checkpoint file.
    """
    path = Path(path)
    if path.is_dir():
        st = path / "model.safetensors"
        if st.is_file():
            from safetensors.torch import load_file
            sd = load_file(str(st))
        else:
            sd = load_checkpoint(path / "pytorch_model.bin", allow_unsafe=allow_unsafe)
    elif path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = load_checkpoint(path, allow_unsafe=allow_unsafe)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]  # fairseq training checkpoint layout
    if "feature_extractor.conv_layers.0.conv.weight" in sd:
        return sd, "transformers"
    if "feature_extractor.conv_layers.0.0.weight" in sd:
        return sd, "fairseq"
    raise ValueError(f"{path}: not a recognised HuBERT checkpoint (unknown key layout)")


def build_content_encoder(path: str | Path, version: str, allow_unsafe: bool = False) -> HubertContentEncoder:
    """Builds the RVC content encoder for `version` ('v1' or 'v2') from a HuBERT checkpoint."""
    if version not in ("v1", "v2"):
        raise ValueError(f"unsupported RVC feature version {version!r}")
    sd, fmt = load_state_dict_any(path, allow_unsafe=allow_unsafe)
    mapped = _map_transformers(sd) if fmt == "transformers" else _map_fairseq(sd)
    output_layer, proj = (9, True) if version == "v1" else (12, False)
    model = HubertContentEncoder(output_layer=output_layer, final_proj=proj)
    if proj and "final_proj.weight" not in mapped:
        raise ValueError("RVC v1 needs final_proj weights, which this HuBERT checkpoint does not contain")
    wanted = set(model.state_dict().keys())
    state = {k: v.float() for k, v in mapped.items() if k in wanted}
    missing = wanted - set(state)
    if missing:
        raise ValueError(f"HuBERT checkpoint is missing weights: {sorted(missing)[:8]}")
    model.load_state_dict(state, strict=True)
    return model.eval()


def compare_with_transformers(model_dir: str | Path, version: str, audio: torch.Tensor) -> torch.Tensor:
    """Runs the upstream-RVC reference (transformers HubertModel) on `audio` [1, N]."""
    from transformers import HubertConfig, HubertModel

    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    config = HubertConfig.from_pretrained(str(model_dir))
    ref = HubertModelWithFinalProj(config)
    sd, fmt = load_state_dict_any(model_dir)
    if fmt != "transformers":
        raise ValueError("compare_with_transformers needs a Transformers-format checkpoint")
    ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    ref = ref.float().eval()
    with torch.no_grad():
        out = ref(input_values=audio, output_hidden_states=True, return_dict=True)
        if version == "v1":
            return ref.final_proj(out.hidden_states[9])
        return out.last_hidden_state
