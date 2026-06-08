"""
Res-TCN-SE-Attention (PyTorch) — faithful reimplementation of the best
subject-dependent model in Nicolescu et al., *Sensors* 2026 (§2.5.1), adapted
to our channel count / sampling rate.

Three branches fused at a dense head:
  • RAW   — Gaussian-noise input → Conv1d stem → 5 depthwise-separable residual
            blocks with Squeeze-and-Excitation (filters 64/96/128/192/256,
            dilations 1/2/4/8/12; first four MaxPool×2) → attention pooling
            over time → dense(192).
  • DWT   — LayerNorm → Dense(256) → Dense(128) on the wavelet vector.
  • FEATS — LayerNorm → Dense(192) → Dense(128) on the spectral/stat vector.
The three embeddings are concatenated → Dense(160) → sigmoid (1 logit out).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis import config as cfg


def _same_pad(kernel: int, dilation: int) -> int:
    return ((kernel - 1) * dilation) // 2


class GaussianNoise(nn.Module):
    """Additive Gaussian noise, active only during training (light augmentation)."""

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class SeparableConv1d(nn.Module):
    """Depthwise (dilated) + pointwise 1×1 convolution, 'same' length."""

    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.depth = nn.Conv1d(in_ch, in_ch, kernel, dilation=dilation,
                               padding=_same_pad(kernel, dilation),
                               groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.point(self.depth(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: GAP → 2-layer gating MLP → channel rescale."""

    def __init__(self, ch):
        super().__init__()
        bottleneck = max(ch // 8, 8)
        self.fc1 = nn.Linear(ch, bottleneck)
        self.fc2 = nn.Linear(bottleneck, ch)

    def forward(self, x):                       # x: (B, C, T)
        s = x.mean(dim=2)                       # squeeze over time
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class ResSepBlock(nn.Module):
    """Two SeparableConv1d (BN-ReLU-SpatialDropout) + SE, residual, opt. pool."""

    def __init__(self, in_ch, out_ch, kernel, dilation, sdrop, pool):
        super().__init__()
        self.conv1 = SeparableConv1d(in_ch, out_ch, kernel, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = SeparableConv1d(out_ch, out_ch, kernel, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout1d(sdrop)         # spatial (whole-channel) dropout
        self.se = SEBlock(out_ch)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pool = nn.MaxPool1d(2) if pool else None

    def forward(self, x):
        res = x if self.proj is None else self.proj(x)
        h = self.drop(F.relu(self.bn1(self.conv1(x))))
        h = self.drop(F.relu(self.bn2(self.conv2(h))))
        h = self.se(h) + res
        if self.pool is not None:
            h = self.pool(h)
        return h


class AttentionPool(nn.Module):
    """1×1 tanh score → 1×1 logit → softmax over time → weighted context."""

    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv1d(ch, ch, 1)
        self.to_logit = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                       # x: (B, C, T)
        a = torch.softmax(self.to_logit(torch.tanh(self.score(x))), dim=2)
        return (x * a).sum(dim=2)               # (B, C)


class RawBranch(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.noise = GaussianNoise(cfg.TCN_NOISE_STD)
        f0 = cfg.TCN_FILTERS[0]
        self.stem = nn.Conv1d(in_ch, f0, cfg.TCN_INIT_KERNEL,
                              padding=_same_pad(cfg.TCN_INIT_KERNEL, 1))
        blocks, prev, n = [], f0, len(cfg.TCN_FILTERS)
        for i, (f, d) in enumerate(zip(cfg.TCN_FILTERS, cfg.TCN_DILATIONS)):
            blocks.append(ResSepBlock(prev, f, cfg.TCN_BLOCK_KERNEL, d,
                                      cfg.TCN_SPATIAL_DROPOUT, pool=i < n - 1))
            prev = f
        self.blocks = nn.ModuleList(blocks)
        self.attn = AttentionPool(prev)
        self.drop = nn.Dropout(cfg.TCN_DROPOUT)
        self.dense = nn.Linear(prev, cfg.TCN_RAW_EMBED)

    def forward(self, x):
        x = self.stem(self.noise(x))
        for b in self.blocks:
            x = b(x)
        h = self.drop(self.attn(x))
        return self.drop(F.relu(self.dense(h)))


class MLPBranch(nn.Module):
    """LayerNorm → Dense(h1) → Dropout → Dense(out), both ReLU."""

    def __init__(self, in_dim, h1, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, h1)
        self.drop = nn.Dropout(cfg.TCN_DROPOUT)
        self.fc2 = nn.Linear(h1, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(self.norm(x)))
        return F.relu(self.fc2(self.drop(x)))


class ResTCNSEAttention(nn.Module):
    """Full 3-branch model. Outputs a single logit (use BCEWithLogitsLoss)."""

    def __init__(self, n_ch, dwt_dim, feats_dim):
        super().__init__()
        self.raw = RawBranch(n_ch)
        self.dwt = MLPBranch(dwt_dim, 256, cfg.TCN_DWT_EMBED)
        self.feats = MLPBranch(feats_dim, cfg.TCN_FEATS_EMBED, 128)
        concat = cfg.TCN_RAW_EMBED + cfg.TCN_DWT_EMBED + 128
        self.head = nn.Linear(concat, cfg.TCN_HEAD_UNITS)
        self.head_drop = nn.Dropout(cfg.TCN_DROPOUT)
        self.out = nn.Linear(cfg.TCN_HEAD_UNITS, 1)

    def forward(self, raw, dwt, feats):
        h = torch.cat([self.raw(raw), self.dwt(dwt), self.feats(feats)], dim=1)
        h = self.head_drop(F.relu(self.head(h)))
        return self.out(h).squeeze(1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
