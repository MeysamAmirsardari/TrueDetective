"""
Shared plumbing for every decoding model.

All models decode the same two classes (secret vs irrelevant, targets excluded)
from the same pooled, cleaned epochs, and write their figures/results into a
per-model sub-folder so the models never clobber one another.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from analysis import config as cfg
from analysis.run_analysis import build_pooled_epochs


def load_binary(verbose: bool = True):
    """Pooled epochs → secret/irrelevant only.

    Returns a dict with the Epochs object plus the arrays every model needs:
    X (n_trials, n_ch, n_times) in volts, y (1=secret, 0=irrelevant), the time
    axis, channel names, and the MNE info (montage attached, for topomaps).
    """
    pooled, report = build_pooled_epochs(verbose=verbose)
    ep = pooled[[cfg.DECODE_POSITIVE, cfg.DECODE_NEGATIVE]]
    X = ep.get_data(copy=True)
    pos_code = cfg.EVENT_ID[cfg.DECODE_POSITIVE]
    y = (ep.events[:, 2] == pos_code).astype(int)
    return {
        "epochs": ep,
        "X": X,
        "y": y,
        "times": ep.times,
        "ch_names": list(ep.ch_names),
        "info": ep.info,
        "report": report,
    }


def stratified_folds(y, n_splits=None, seed=None):
    """Stratified K-fold splitter (preserves the secret/irrelevant ratio)."""
    n_splits = n_splits or cfg.DECODE_CV_FOLDS
    seed = cfg.DECODE_SEED if seed is None else seed
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def zscore_per_trial(X):
    """Per-trial, per-channel z-scoring (the paper's window standardisation)."""
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mu) / sd


def out_dirs(model_name: str):
    """Create and return (figures_dir, results_dir) for one model."""
    fig = cfg.FIGURES_DIR / model_name
    res = cfg.RESULTS_DIR / model_name
    fig.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    return fig, res
