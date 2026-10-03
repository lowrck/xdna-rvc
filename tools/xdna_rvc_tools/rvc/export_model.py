"""Inference-only RVC synthesizer for ONNX export at fixed shapes.

Differences from the upstream training/inference module (all numerically neutral,
verified against `upstream.models` by the conversion tool and the test suite):

* Fixed streaming geometry. Frames T, skip_head and return_length are export-time
  constants (RVC realtime semantics: flow_head = max(skip_head - 24, 0); only
  return_length frames are decoded).
* No masks. With a fixed, fully valid window every mask is all-ones, so masking ops
  are removed instead of being exported as Where/Mul nodes.
* Relative-position attention uses constant gather indices instead of the
  pad/reshape skew trick, giving a static, O(T * window) graph.
* Randomness is explicit: the flow prior noise `rnd` and the NSF excitation noise
  `noise` are model inputs. This makes PyTorch-vs-ONNX validation deterministic and
  avoids RandomNormalLike ops (which accelerators generally do not support).
* Weight norm is folded into plain weights.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import RvcCheckpointInfo, build_upstream_synthesizer

FLOW_MARGIN = 24  # upstream: flow_head = max(skip_head - 24, 0)


@dataclass(frozen=True)
class StreamGeometry:
    frames: int          # T: generator input frames at 100 Hz
    skip_head: int       # frames of left context not decoded
    return_length: int   # frames decoded

    @property
    def flow_head(self) -> int:
        return max(self.skip_head - FLOW_MARGIN, 0)

    @property
    def dec_head(self) -> int:
        return self.skip_head - self.flow_head

    @property
    def flow_frames(self) -> int:
        return self.frames - self.flow_head

    def validate(self) -> None:
        if self.frames <= 0 or self.return_length <= 0 or self.skip_head < 0:
            raise ValueError(f"invalid geometry {self}")
        if self.skip_head + self.return_length > self.frames:
            raise ValueError(f"skip_head + return_length exceeds frames: {self}")

    def tag(self) -> str:
        return f"T{self.frames}_s{self.skip_head}_r{self.return_length}"


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels of a [B, C, T] tensor (upstream modules.LayerNorm)."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.gamma = nn.Parameter(src.gamma.detach().clone())
        self.beta = nn.Parameter(src.beta.detach().clone())
        self.eps = src.eps
        self.channels = int(self.gamma.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class RelativeSelfAttention(nn.Module):
    """VITS relative-position self-attention (window 10, shared heads), static T."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.conv_q, self.conv_k, self.conv_v, self.conv_o = (copy.deepcopy(m) for m in
                                                              (src.conv_q, src.conv_k, src.conv_v, src.conv_o))
        self.n_heads = src.n_heads
        self.k_channels = src.k_channels
        self.window = src.window_size
        self.emb_rel_k = nn.Parameter(src.emb_rel_k.detach().clone())  # [1, 2w+1, d]
        self.emb_rel_v = nn.Parameter(src.emb_rel_v.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        h, d, w = self.n_heads, self.k_channels, self.window
        q = self.conv_q(x).view(b, h, d, t).transpose(2, 3) / math.sqrt(d)   # [b,h,t,d]
        k = self.conv_k(x).view(b, h, d, t).transpose(2, 3)
        v = self.conv_v(x).view(b, h, d, t).transpose(2, 3)
        scores = torch.matmul(q, k.transpose(-2, -1))                           # [b,h,t,t]

        # Constant index tables (t is static at export time).
        i = torch.arange(t).unsqueeze(1)
        j = torch.arange(t).unsqueeze(0)
        rel = j - i + w                                                         # [t,t] in [w-t+1, w+t-1]
        band = ((rel >= 0) & (rel <= 2 * w)).to(x.dtype)
        rel_idx = rel.clamp(0, 2 * w).expand(b, h, t, t)
        qe = torch.matmul(q, self.emb_rel_k.transpose(-2, -1))                 # [b,h,t,2w+1]
        scores = scores + torch.gather(qe, 3, rel_idx) * band

        p = torch.softmax(scores, dim=-1)
        out = torch.matmul(p, v)                                                # [b,h,t,d]
        r = torch.arange(2 * w + 1).unsqueeze(0)                                # [1,2w+1]
        col = torch.arange(t).unsqueeze(1) + r - w                              # [t,2w+1] absolute j
        valid = ((col >= 0) & (col < t)).to(x.dtype)
        col_idx = col.clamp(0, t - 1).expand(b, h, t, 2 * w + 1)
        rel_w = torch.gather(p, 3, col_idx) * valid                             # [b,h,t,2w+1]
        out = out + torch.matmul(rel_w, self.emb_rel_v)
        out = out.transpose(2, 3).reshape(b, c, t)
        return self.conv_o(out)


class FeedForward(nn.Module):
    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.conv_1 = copy.deepcopy(src.conv_1)
        self.conv_2 = copy.deepcopy(src.conv_2)
        k = src.kernel_size
        self.pad = ((k - 1) // 2, k // 2)
        if src.causal or src.activation is not None:
            raise ValueError("only the standard RVC FFN (same padding, ReLU) is supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv_1(F.pad(x, self.pad)))
        return self.conv_2(F.pad(x, self.pad))


class TextEncoder(nn.Module):
    def __init__(self, src: nn.Module, flow_head: int) -> None:
        super().__init__()
        self.emb_phone = copy.deepcopy(src.emb_phone)
        self.emb_pitch = copy.deepcopy(src.emb_pitch) if hasattr(src, "emb_pitch") else None
        enc = src.encoder
        self.attn = nn.ModuleList(RelativeSelfAttention(a) for a in enc.attn_layers)
        self.norm1 = nn.ModuleList(ChannelLayerNorm(n) for n in enc.norm_layers_1)
        self.ffn = nn.ModuleList(FeedForward(f) for f in enc.ffn_layers)
        self.norm2 = nn.ModuleList(ChannelLayerNorm(n) for n in enc.norm_layers_2)
        self.proj = copy.deepcopy(src.proj)
        self.out_channels = src.out_channels
        self.scale = math.sqrt(src.hidden_channels)
        self.flow_head = flow_head

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor | None):
        x = self.emb_phone(phone)
        if self.emb_pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = F.leaky_relu(x * self.scale, 0.1).transpose(1, 2)  # [b, hidden, t]
        for attn, n1, ffn, n2 in zip(self.attn, self.norm1, self.ffn, self.norm2):
            x = n1(x + attn(x))
            x = n2(x + ffn(x))
        x = x[:, :, self.flow_head:]
        m, logs = torch.split(self.proj(x), self.out_channels, dim=1)
        return m, logs


class WaveNet(nn.Module):
    """Mask-free copy of upstream modules.WN (weight norm already removed)."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.in_layers = copy.deepcopy(src.in_layers)
        self.res_skip_layers = copy.deepcopy(src.res_skip_layers)
        self.cond_layer = copy.deepcopy(src.cond_layer)
        self.hidden = src.hidden_channels
        self.n_layers = src.n_layers

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        g_all = self.cond_layer(g)
        hc = self.hidden
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g_all[:, i * 2 * hc:(i + 1) * 2 * hc, :]
            a = x_in + g_l
            acts = torch.tanh(a[:, :hc]) * torch.sigmoid(a[:, hc:])
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = x + rs[:, :hc]
                out = out + rs[:, hc:]
            else:
                out = out + rs
        return out


class FlowReverse(nn.Module):
    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        layers = [f for f in src.flows if f.__class__.__name__ == "ResidualCouplingLayer"]
        if len(layers) * 2 != len(src.flows):
            raise ValueError("unexpected flow layout (expected alternating coupling/flip)")
        for f in layers:
            if not f.mean_only:
                raise ValueError("only mean-only coupling layers are supported")
        self.pre = nn.ModuleList(copy.deepcopy(f.pre) for f in layers)
        self.enc = nn.ModuleList(WaveNet(f.enc) for f in layers)
        self.post = nn.ModuleList(copy.deepcopy(f.post) for f in layers)
        self.half = layers[0].half_channels

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # upstream: for flow in reversed([C0, F, C1, F, C2, F, C3, F]): x = flow(x, reverse=True)
        for idx in reversed(range(len(self.pre))):
            x = torch.flip(x, [1])
            x0, x1 = x[:, :self.half], x[:, self.half:]
            h = self.enc[idx](self.pre[idx](x0), g)
            x = torch.cat([x0, x1 - self.post[idx](h)], dim=1)
        return x


class NsfSource(nn.Module):
    """Upstream SineGen (yxlllc frame-rate phase) + SourceModuleHnNSF, noise as input."""

    def __init__(self, src: nn.Module, upp: int) -> None:
        super().__init__()
        sg = src.l_sin_gen
        if sg.harmonic_num != 0:
            raise ValueError("only harmonic_num=0 NSF sources are supported")
        self.sr = float(sg.sampling_rate)
        self.sine_amp = sg.sine_amp
        self.noise_std = sg.noise_std
        self.threshold = sg.voiced_threshold
        self.upp = upp
        self.l_linear = copy.deepcopy(src.l_linear)

    def forward(self, f0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # f0: [b, L] Hz; noise: [b, L*upp, 1] standard normal -> har_source [b, 1, L*upp]
        b, n = f0.shape
        f0 = f0.unsqueeze(-1)                                                  # [b,L,1]
        steps = torch.arange(1, self.upp + 1, dtype=f0.dtype).view(1, 1, -1)
        rad = f0 / self.sr * steps                                             # [b,L,upp]
        rad2 = torch.fmod(rad[:, :, -1:] + 0.5, 1.0) - 0.5
        rad_acc = torch.fmod(torch.cumsum(rad2, dim=1), 1.0)
        rad = rad + torch.cat([torch.zeros_like(rad_acc[:, :1]), rad_acc[:, :-1]], dim=1)
        rad = rad.reshape(b, n * self.upp, 1)
        sine = torch.sin(2 * math.pi * rad) * self.sine_amp
        uv = (f0 > self.threshold).to(f0.dtype)                                # [b,L,1]
        uv = uv.expand(b, n, self.upp).reshape(b, n * self.upp, 1)             # nearest upsampling
        noise_amp = uv * self.noise_std + (1 - uv) * (self.sine_amp / 3)
        sine = sine * uv + noise_amp * noise
        return torch.tanh(self.l_linear(sine)).transpose(1, 2)


class Decoder(nn.Module):
    """HiFi-GAN (optionally NSF) decoder sharing upstream weights."""

    def __init__(self, src: nn.Module, nsf: bool) -> None:
        super().__init__()
        self.conv_pre = copy.deepcopy(src.conv_pre)
        self.ups = copy.deepcopy(src.ups)
        self.resblocks = copy.deepcopy(src.resblocks)
        self.conv_post = copy.deepcopy(src.conv_post)
        self.cond = copy.deepcopy(src.cond)
        self.num_kernels = src.num_kernels
        self.nsf = nsf
        if nsf:
            self.noise_convs = copy.deepcopy(src.noise_convs)
            self.source = NsfSource(src.m_source, src.upp)
            self.slope = src.lrelu_slope
        else:
            self.slope = 0.1  # modules.LRELU_SLOPE

    def forward(self, x: torch.Tensor, g: torch.Tensor, f0: torch.Tensor | None = None,
                noise: torch.Tensor | None = None) -> torch.Tensor:
        har = self.source(f0, noise) if self.nsf else None
        x = self.conv_pre(x) + self.cond(g)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, self.slope))
            if self.nsf:
                x = x + self.noise_convs[i](har)
            xs = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels
        x = F.leaky_relu(x)  # default slope 0.01, as upstream
        return torch.tanh(self.conv_post(x))


class GeneratorExport(nn.Module):
    """Fixed-geometry RVC synthesizer.

    f0 models:    (phone[1,T,D], pitch[1,T] int64, pitchf[1,T] f32, sid[1] int64,
                   rnd[1,C,T-flow_head] f32, noise[1,R*upp,1] f32) -> audio[1, R*upp]
    no-f0 models: (phone, sid, rnd) -> audio
    """

    def __init__(self, info: RvcCheckpointInfo, state_dict: dict, geom: StreamGeometry) -> None:
        super().__init__()
        geom.validate()
        net = build_upstream_synthesizer(info, state_dict)
        net.remove_weight_norm()
        self.geom = geom
        self.uses_f0 = info.uses_f0
        self.upp = info.upsample_factor
        self.emb_g = copy.deepcopy(net.emb_g)
        self.enc_p = TextEncoder(net.enc_p, geom.flow_head)
        self.flow = FlowReverse(net.flow)
        self.dec = Decoder(net.dec, nsf=info.uses_f0)
        self.inter_channels = info.config["inter_channels"]

    def input_shapes(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        g = self.geom
        shapes = {"phone": ((1, g.frames, self.enc_p.emb_phone.in_features), torch.float32)}
        if self.uses_f0:
            shapes["pitch"] = ((1, g.frames), torch.int64)
            shapes["pitchf"] = ((1, g.frames), torch.float32)
        shapes["sid"] = ((1,), torch.int64)
        shapes["rnd"] = ((1, self.inter_channels, g.flow_frames), torch.float32)
        if self.uses_f0:
            shapes["noise"] = ((1, g.return_length * self.upp, 1), torch.float32)
        return shapes

    def forward(self, *args):
        if self.uses_f0:
            phone, pitch, pitchf, sid, rnd, noise = args
        else:
            phone, sid, rnd = args
            pitch = pitchf = noise = None
        g = self.geom
        spk = self.emb_g(sid).unsqueeze(-1)
        m, logs = self.enc_p(phone, pitch)
        z = self.flow(m + torch.exp(logs) * rnd * 0.66666, spk)
        z = z[:, :, g.dec_head:g.dec_head + g.return_length]
        f0 = pitchf[:, g.skip_head:g.skip_head + g.return_length] if self.uses_f0 else None
        audio = self.dec(z, spk, f0, noise)
        return audio.squeeze(1)


def reference_infer(upstream_net: nn.Module, info: RvcCheckpointInfo, geom: StreamGeometry,
                    inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Runs the *upstream* synthesizer's infer() on the same inputs, injecting the given
    noise tensors where upstream calls torch.randn_like."""
    queue = [inputs["rnd"]] + ([inputs["noise"]] if info.uses_f0 else [])
    original = torch.randn_like

    def fake_randn_like(t, *a, **k):
        if not queue:
            raise RuntimeError("unexpected extra randn_like call in upstream infer()")
        n = queue.pop(0)
        if tuple(n.shape) != tuple(t.shape):
            raise RuntimeError(f"noise shape {tuple(n.shape)} != requested {tuple(t.shape)}")
        return n.to(t.dtype)

    lengths = torch.tensor([geom.frames], dtype=torch.long)
    torch.randn_like = fake_randn_like
    try:
        with torch.no_grad():
            if info.uses_f0:
                o, _, _ = upstream_net.infer(inputs["phone"], lengths, inputs["pitch"], inputs["pitchf"],
                                             inputs["sid"], geom.skip_head, geom.return_length)
            else:
                o, _, _ = upstream_net.infer(inputs["phone"], lengths, inputs["sid"], geom.skip_head,
                                             geom.return_length)
    finally:
        torch.randn_like = original
    if queue:
        raise RuntimeError("upstream infer() consumed fewer noise tensors than expected")
    return o.squeeze(1)


def example_inputs(model: GeneratorExport, seed: int = 0, speech_like: bool = True) -> dict[str, torch.Tensor]:
    """Deterministic inputs with realistic value ranges."""
    gen = torch.Generator().manual_seed(seed)
    shapes = model.input_shapes()
    pitchf = None
    if "pitchf" in shapes:
        n = shapes["pitchf"][0][1]
        pitchf = 180 + 60 * torch.sin(torch.linspace(0, 6.28, n))
        pitchf[: n // 10] = 0.0  # an unvoiced stretch
        pitchf = pitchf.view(1, n)
    out = {}
    for name, (shape, dtype) in shapes.items():
        if name == "phone":
            out[name] = torch.randn(shape, generator=gen) * (0.5 if speech_like else 1.0)
        elif name == "pitchf":
            out[name] = pitchf
        elif name == "pitch":
            out[name] = coarse_pitch(pitchf)
        elif name == "sid":
            out[name] = torch.zeros(shape, dtype=torch.int64)
        else:
            out[name] = torch.randn(shape, generator=gen)
    return out


def coarse_pitch(f0: torch.Tensor) -> torch.Tensor:
    """RVC f0 -> coarse pitch bin (1..255), identical to upstream get_f0_post()."""
    f0_min, f0_max = 50.0, 1100.0
    mel_min = 1127 * math.log(1 + f0_min / 700)
    mel_max = 1127 * math.log(1 + f0_max / 700)
    mel = 1127 * torch.log(1 + f0 / 700)
    mel = torch.where(mel > 0, (mel - mel_min) * 254 / (mel_max - mel_min) + 1, mel)
    mel = mel.clamp(1, 255)
    return torch.round(mel).long()
