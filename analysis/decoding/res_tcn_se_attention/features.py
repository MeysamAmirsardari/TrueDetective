"""
Hand-crafted feature branches for Res-TCN-SE-Attention.

Two compact descriptor vectors are computed per (z-scored) epoch, mirroring the
paper's DWT and FEATS branches but sized for our 28-channel / 250 Hz montage:

  • DWT   — 4-level db4 wavelet decomposition per channel. From each of the 5
            sub-bands (cA4, cD4..cD1): 6 summary stats (min, max, median, mean,
            std, var) + relative energy → 35 values/channel.

  • FEATS — extended-band spectral/statistical descriptors per channel: 11 band
            powers, 11 relative band powers, θ/β and (θ+α)/β ratios, spectral
            entropy, Hjorth activity/mobility/complexity, and 5 band-limited
            differential entropies → 33 values/channel, plus 2 frontal
            asymmetry indices (α and β) from a right/left frontal pair.

Dimensions scale with channel count (paper: 175 / 167 for 5 channels; here
~980 / ~926 for 28), which the model's branch MLPs adapt to automatically.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import welch

from analysis import config as cfg

# Extended frequency bands (Hz). Note our recordings are low-passed at 30 Hz,
# so the γ bands carry little power — kept for fidelity to the paper's vector.
_BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "alpha_low": (8, 10), "alpha_high": (10, 13),
    "beta": (13, 30), "beta_low": (13, 20), "beta_high": (20, 30),
    "gamma": (30, 45), "gamma_low": (30, 38), "gamma_middle": (38, 45),
}
_DE_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
# Right/left frontal pairs tried in order for the asymmetry indices.
_FRONTAL_PAIRS = [("F4", "F3"), ("Fp2", "Fp1"), ("F8", "F7"), ("AF4", "AF3")]
_EPS = 1e-10


# ── DWT branch ─────────────────────────────────────────────────────────
def _dwt_channel(sig: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec(sig, cfg.DWT_WAVELET, level=cfg.DWT_LEVEL)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total = energies.sum() + _EPS
    feats = []
    for c, e in zip(coeffs, energies):
        feats += [c.min(), c.max(), np.median(c), c.mean(), c.std(), c.var(),
                  e / total]
    return np.asarray(feats, dtype=np.float32)        # 7 * (level+1) = 35


def dwt_vector(epoch: np.ndarray) -> np.ndarray:
    """epoch (C, T) → flat DWT feature vector (C * 35)."""
    return np.concatenate([_dwt_channel(epoch[c]) for c in range(epoch.shape[0])])


# ── FEATS branch ───────────────────────────────────────────────────────
def _bandpower(psd, freqs, lo, hi):
    idx = (freqs >= lo) & (freqs <= hi)
    if not idx.any():
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))


def _hjorth(sig):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    v0 = sig.var() + _EPS
    v1 = d1.var() + _EPS
    v2 = d2.var() + _EPS
    activity = v0
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / (mobility + _EPS)
    return activity, mobility, complexity


def _feats_channel(sig, sfreq):
    nperseg = int(min(len(sig), 256))
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg, window="hann",
                       detrend="constant")
    total = float(np.trapz(psd, freqs)) + _EPS

    bp = {b: _bandpower(psd, freqs, lo, hi) for b, (lo, hi) in _BANDS.items()}
    powers = [bp[b] for b in _BANDS]                         # 11
    rel = [p / total for p in powers]                        # 11
    theta_beta = bp["theta"] / (bp["beta"] + _EPS)
    ta_beta = (bp["theta"] + bp["alpha"]) / (bp["beta"] + _EPS)

    p_norm = psd / (psd.sum() + _EPS)
    spec_ent = float(-np.sum(p_norm * np.log(p_norm + _EPS)) / np.log(len(p_norm)))

    activity, mobility, complexity = _hjorth(sig)
    de = [0.5 * np.log(2 * np.pi * np.e * (bp[b] + _EPS)) for b in _DE_BANDS]  # 5

    return np.asarray(
        powers + rel + [theta_beta, ta_beta, spec_ent,
                        activity, mobility, complexity] + de,
        dtype=np.float32,
    )                                                        # 33


def _frontal_asymmetry(epoch, sfreq, ch_names):
    """log(α_R) − log(α_L) and log(β_R) − log(β_L) for the first available pair."""
    idx = {c: i for i, c in enumerate(ch_names)}
    for right, left in _FRONTAL_PAIRS:
        if right in idx and left in idx:
            out = []
            for band in ("alpha", "beta"):
                lo, hi = _BANDS[band]
                vals = []
                for ch in (right, left):
                    f, p = welch(epoch[idx[ch]], fs=sfreq,
                                 nperseg=int(min(epoch.shape[1], 256)),
                                 window="hann", detrend="constant")
                    vals.append(np.log(_bandpower(p, f, lo, hi) + _EPS))
                out.append(vals[0] - vals[1])
            return np.asarray(out, dtype=np.float32)
    return np.zeros(2, dtype=np.float32)


def feats_vector(epoch: np.ndarray, sfreq: float, ch_names) -> np.ndarray:
    """epoch (C, T) → flat FEATS vector (C * 33 + 2)."""
    per_ch = np.concatenate([_feats_channel(epoch[c], sfreq)
                             for c in range(epoch.shape[0])])
    return np.concatenate([per_ch, _frontal_asymmetry(epoch, sfreq, ch_names)])


# ── Batch extraction ───────────────────────────────────────────────────
def extract_all(X: np.ndarray, sfreq: float, ch_names):
    """X (N, C, T) z-scored → (dwt (N, Ddwt), feats (N, Dfeats)) float32."""
    dwt = np.stack([dwt_vector(x) for x in X]).astype(np.float32)
    feats = np.stack([feats_vector(x, sfreq, ch_names) for x in X]).astype(np.float32)
    return dwt, feats
