"""
Time-resolved decoding (MVPA): secret vs irrelevant — linear baseline.

A SlidingEstimator fits one standardised, class-balanced logistic regression
per time sample (chance AUC = 0.5). Produces:
    1. decoding-over-time curve (all good channels),
    2. per-channel decodability ranking + scalp topomap,
    3. channel × time AUC heatmap.

This is the fast linear reference the heavier models are compared against.

Run:  python -m analysis.decoding.sliding_logreg.run
"""

from __future__ import annotations

# Allow direct execution (python analysis/decoding/sliding_logreg/run.py) as
# well as module execution (python -m analysis.decoding.sliding_logreg.run).
import pathlib as _pl
import sys as _sys
if __package__ in (None, ""):
    for _root in _pl.Path(__file__).resolve().parents:
        if (_root / "analysis" / "__init__.py").exists():
            _sys.path.insert(0, str(_root))
            break

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import config as cfg
from analysis.decoding import common

MODEL = "sliding_logreg"


# ── Estimators ─────────────────────────────────────────────────────────
def _clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


# ── Analyses ───────────────────────────────────────────────────────────
def decode_over_time(X, y):
    sl = SlidingEstimator(_clf(), scoring=cfg.DECODE_SCORING, n_jobs=-1,
                          verbose=False)
    scores = cross_val_multiscore(sl, X, y, cv=common.stratified_folds(y),
                                  n_jobs=1)
    return scores.mean(0), scores.std(0)


def decode_per_channel(X, y, times, ch_names):
    lo, hi = cfg.DECODE_CHANNEL_WINDOW
    mask = (times >= lo) & (times <= hi)
    out: dict[str, float] = {}
    for ci, ch in enumerate(ch_names):
        Xc = X[:, ci, :][:, mask]
        auc = cross_val_score(_clf(), Xc, y, cv=common.stratified_folds(y),
                              scoring=cfg.DECODE_SCORING, n_jobs=-1)
        out[ch] = float(auc.mean())
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def decode_channel_time(X, y):
    n_ch, n_t = X.shape[1], X.shape[2]
    amap = np.empty((n_ch, n_t))
    for ci in range(n_ch):
        sl = SlidingEstimator(_clf(), scoring=cfg.DECODE_SCORING, n_jobs=-1,
                              verbose=False)
        sc = cross_val_multiscore(sl, X[:, ci:ci + 1, :], y,
                                  cv=common.stratified_folds(y), n_jobs=1)
        amap[ci] = sc.mean(0)
    return amap


# ── Figures ────────────────────────────────────────────────────────────
def plot_time(times, mean, std, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, mean, color="C3", lw=2, label="AUC (all channels)")
    ax.fill_between(times, mean - std, mean + std, color="C3", alpha=0.2,
                    label="±1 SD across folds")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Time-resolved decoding: secret vs irrelevant")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_channel_ranking(ch_auc, path):
    chs = list(ch_auc)[::-1]
    vals = [ch_auc[c] for c in chs]
    fig, ax = plt.subplots(figsize=(6, max(4, 0.28 * len(chs))))
    colors = ["C3" if v >= 0.5 else "C0" for v in vals]
    ax.barh(chs, vals, color=colors)
    ax.axvline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Per-channel decodability (secret vs irrelevant)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_channel_topomap(ch_auc, info, path):
    vals = np.array([ch_auc[ch] for ch in info["ch_names"]])
    try:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False,
                                     cmap="Reds", vlim=(0.5, float(vals.max())))
        fig.colorbar(im, ax=ax, label="ROC-AUC")
        ax.set_title("Decodability topography")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        mne.utils.warn(f"decoding topomap skipped: {exc}")


def plot_channel_time(amap, times, ch_names, path):
    order = np.argsort(amap.max(axis=1))
    amap_o = amap[order]
    chs_o = [ch_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.26 * len(chs_o))))
    im = ax.imshow(amap_o, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=0.5 - (amap.max() - 0.5), vmax=amap.max(),
                   extent=[times[0], times[-1], 0, len(chs_o)])
    ax.set_yticks(np.arange(len(chs_o)) + 0.5)
    ax.set_yticklabels(chs_o, fontsize=6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_title("Channel × time decoding AUC")
    fig.colorbar(im, ax=ax, label="ROC-AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    fig_dir, res_dir = common.out_dirs(MODEL)
    data = common.load_binary(verbose=True)
    X, y, times, ch_names, info = (data["X"], data["y"], data["times"],
                                   data["ch_names"], data["info"])
    print(f"\nDecoding {cfg.DECODE_POSITIVE} vs {cfg.DECODE_NEGATIVE} "
          f"(targets excluded): {int((y == 1).sum())} secret vs "
          f"{int((y == 0).sum())} irrelevant, {len(ch_names)} channels.")

    mean_t, std_t = decode_over_time(X, y)
    peak_i = int(np.argmax(mean_t))
    print(f"\nTime-resolved (all channels): peak AUC={mean_t[peak_i]:.3f} "
          f"at {times[peak_i]*1000:.0f} ms")
    plot_time(times, mean_t, std_t, fig_dir / "decoding_time.png")

    ch_auc = decode_per_channel(X, y, times, ch_names)
    top = list(ch_auc.items())[:cfg.DECODE_TOP_K]
    print(f"\nMost decodable channels (window {cfg.DECODE_CHANNEL_WINDOW[0]}–"
          f"{cfg.DECODE_CHANNEL_WINDOW[1]}s):")
    for ch, a in top:
        print(f"   {ch:5s} AUC={a:.3f}")
    plot_channel_ranking(ch_auc, fig_dir / "decoding_channels.png")
    plot_channel_topomap(ch_auc, info, fig_dir / "decoding_topomap.png")

    amap = decode_channel_time(X, y)
    plot_channel_time(amap, times, ch_names, fig_dir / "decoding_channel_time.png")

    print(f"\nFigures written to {fig_dir}")
    out = {
        "model": MODEL,
        "classes": {cfg.DECODE_POSITIVE: 1, cfg.DECODE_NEGATIVE: 0},
        "n_secret": int((y == 1).sum()),
        "n_irrelevant": int((y == 0).sum()),
        "n_channels": len(ch_names),
        "scoring": cfg.DECODE_SCORING,
        "cv_folds": cfg.DECODE_CV_FOLDS,
        "time_resolved": {
            "peak_auc": float(mean_t[peak_i]),
            "peak_time_s": float(times[peak_i]),
            "times_s": [float(t) for t in times],
            "auc_mean": [float(v) for v in mean_t],
            "auc_std": [float(v) for v in std_t],
        },
        "per_channel_auc": ch_auc,
        "top_channels": [c for c, _ in top],
        "channel_window_s": list(cfg.DECODE_CHANNEL_WINDOW),
    }
    (res_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"Results written to {res_dir / 'results.json'}")


if __name__ == "__main__":
    main()
