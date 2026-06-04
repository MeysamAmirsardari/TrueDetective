"""
ERP measurement and CIT detection statistics.

The CIT logic: a guilty participant's brain treats the concealed *secret* like
a rare, meaningful target, so the secret should evoke a larger P300 than the
*irrelevant* items. We quantify that two ways, both on the parietal midline:

    1.  Bootstrapped mean-amplitude difference (Rosenfeld-style).
        Resample epochs with replacement; the fraction of bootstrap iterations
        where secret-amplitude exceeds irrelevant-amplitude is a one-sided
        detection confidence. Secret "detected" if that fraction ≥ 1-alpha.

    2.  Cluster-based permutation test over the whole post-stimulus window.
        Guards against cherry-picking the 300-600 ms window: finds
        time-clusters where secret ≠ irrelevant with family-wise error control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np

from . import config as cfg


@dataclass
class P300Result:
    channel: str
    secret_uv: float
    irrelevant_uv: float
    target_uv: float
    diff_uv: float                       # secret - irrelevant
    boot_p: float                        # one-sided p (secret ≤ irrelevant)
    boot_ci: tuple[float, float]         # 95% CI of the difference (µV)
    detected: bool
    n_secret: int
    n_irrelevant: int
    n_target: int
    cluster_p: list[float] = field(default_factory=list)
    cluster_windows: list[tuple[float, float]] = field(default_factory=list)


def _pick_channel(epochs: mne.Epochs) -> str:
    """Preferred P300 site, falling back through the secondary midline list."""
    if cfg.P300_CHANNEL in epochs.ch_names:
        return cfg.P300_CHANNEL
    for ch in cfg.P300_SECONDARY:
        if ch in epochs.ch_names:
            return ch
    return epochs.ch_names[0]


def _window_mean(epochs: mne.Epochs, ch: str) -> np.ndarray:
    """Per-epoch mean amplitude (µV) in the P300 window for one channel."""
    e = epochs.copy().pick([ch])
    data = e.get_data(copy=False)[:, 0, :]            # epochs x time, volts
    times = e.times
    mask = (times >= cfg.P300_WINDOW[0]) & (times <= cfg.P300_WINDOW[1])
    return data[:, mask].mean(axis=1) * cfg.MICROVOLT


def _bootstrap_diff(secret: np.ndarray, irr: np.ndarray):
    """Bootstrap the secret-minus-irrelevant mean-amplitude difference.

    Returns (observed_diff, one_sided_p, (ci_lo, ci_hi)). The one-sided p is
    the bootstrap probability that secret ≤ irrelevant — small p ⇒ the secret
    P300 is reliably larger, i.e. concealed information is detected.
    """
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    n_p, n_i = len(secret), len(irr)
    obs = secret.mean() - irr.mean()

    diffs = np.empty(cfg.BOOTSTRAP_ITERS)
    for k in range(cfg.BOOTSTRAP_ITERS):
        bp = secret[rng.integers(0, n_p, n_p)]
        bi = irr[rng.integers(0, n_i, n_i)]
        diffs[k] = bp.mean() - bi.mean()

    p_one_sided = float(np.mean(diffs <= 0.0))
    ci = (float(np.percentile(diffs, 2.5)),
          float(np.percentile(diffs, 97.5)))
    return float(obs), p_one_sided, ci


def _cluster_test(epochs: mne.Epochs, ch: str):
    """1-D cluster-based permutation test, secret vs irrelevant, one channel.

    Scans the whole *post-stimulus* time-course (strictly t > 0) so significant
    effects aren't restricted to the a-priori P300 window, while excluding the
    baseline and the onset sample itself — the t = 0 point is the last baseline
    sample, so after baseline-correction its point-wise difference is
    degenerate and would surface as a spurious single-sample edge cluster.
    Returns the significant clusters' p-values and (start, end) times.
    """
    e = epochs.copy().pick([ch])
    post = e.times > 0.0                              # drop baseline + onset
    times = e.times[post]
    secret = e["secret"].get_data(copy=False)[:, 0, post] * cfg.MICROVOLT
    irr = e["irrelevant"].get_data(copy=False)[:, 0, post] * cfg.MICROVOLT

    T_obs, clusters, cluster_pv, _ = mne.stats.permutation_cluster_test(
        [secret, irr],
        n_permutations=cfg.CLUSTER_PERMUTATIONS,
        seed=cfg.CLUSTER_SEED,
        tail=0,
        out_type="mask",
        verbose="ERROR",
    )

    sig_p, sig_win = [], []
    for mask, pv in zip(clusters, cluster_pv):
        if pv >= cfg.DETECTION_ALPHA:
            continue
        idx = np.where(mask)[0]
        lo, hi = float(times[idx[0]]), float(times[idx[-1]])
        # Discard sub-component blips: a real ERP effect is sustained, not a
        # one- or two-sample flicker at the resolution limit.
        if (hi - lo) < cfg.CLUSTER_MIN_DURATION:
            continue
        sig_p.append(float(pv))
        sig_win.append((lo, hi))
    return sig_p, sig_win


def analyze_p300(epochs: mne.Epochs) -> P300Result:
    """Full P300/CIT analysis on pooled epochs."""
    ch = _pick_channel(epochs)

    secret_amp = _window_mean(epochs["secret"], ch)
    irr_amp = _window_mean(epochs["irrelevant"], ch)
    has_target = "target" in epochs.event_id and len(epochs["target"]) > 0
    target_amp = _window_mean(epochs["target"], ch) if has_target else np.array([])

    obs, boot_p, ci = _bootstrap_diff(secret_amp, irr_amp)
    cluster_p, cluster_win = _cluster_test(epochs, ch)

    return P300Result(
        channel=ch,
        secret_uv=float(secret_amp.mean()),
        irrelevant_uv=float(irr_amp.mean()),
        target_uv=float(target_amp.mean()) if target_amp.size else float("nan"),
        diff_uv=obs,
        boot_p=boot_p,
        boot_ci=ci,
        detected=boot_p < cfg.DETECTION_ALPHA,
        n_secret=len(secret_amp),
        n_irrelevant=len(irr_amp),
        n_target=int(target_amp.size),
        cluster_p=cluster_p,
        cluster_windows=cluster_win,
    )


def evokeds_by_condition(epochs: mne.Epochs) -> dict[str, mne.Evoked]:
    """Grand-average ERP per condition, for plotting."""
    out = {}
    for cond in cfg.EVENT_ID:
        if cond in epochs.event_id and len(epochs[cond]) > 0:
            out[cond] = epochs[cond].average()
    return out
