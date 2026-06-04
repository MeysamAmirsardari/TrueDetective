"""
Epoching: cleaned Raw + events → baseline-corrected, artifact-rejected Epochs.

We epoch each session separately (each has its own dropped channels and ICA
solution) and then concatenate. Because A2 may be dropped from one session but
not another, we intersect channel sets before concatenating so the pooled
Epochs object is rectangular.
"""

from __future__ import annotations

import mne
import numpy as np

from . import config as cfg
from .io_xdf import SessionData
from .preprocess import PreprocResult


def make_epochs(raw: mne.io.BaseRaw, events: np.ndarray) -> mne.Epochs:
    """Epoch one session around stimulus markers.

    Only probe/irrelevant/target codes are kept; control markers
    (session/block start, cues) are ignored by passing an explicit event_id.
    Peak-to-peak rejection is in volts, so convert the µV threshold.
    """
    # Drop events whose codes aren't in EVENT_ID (control markers 10/11/12/99).
    keep = np.isin(events[:, 2], list(cfg.EVENT_ID.values()))
    events = events[keep]

    reject = {"eeg": cfg.REJECT_UV * 1e-6}
    epochs = mne.Epochs(
        raw, events, event_id=cfg.EVENT_ID,
        tmin=cfg.TMIN, tmax=cfg.TMAX,
        baseline=cfg.BASELINE,
        reject=reject,
        decim=cfg.DECIM,
        preload=True,
        verbose="ERROR",
    )
    return epochs


def build_session_epochs(session: SessionData,
                         pre: PreprocResult) -> mne.Epochs:
    """Convenience wrapper: epoch a preprocessed session."""
    ep = make_epochs(pre.raw, session.events)
    ep.metadata = None
    return ep


def pool_epochs(epochs_list: list[mne.Epochs]) -> mne.Epochs:
    """Concatenate per-session Epochs onto their common channel set.

    Sessions can differ in which channels were dropped as bad, so we restrict
    every Epochs object to the intersection of channels (preserving order)
    before concatenating. This keeps the pooled ERP free of channels that were
    only clean in one session.
    """
    if len(epochs_list) == 1:
        return epochs_list[0]

    common = set(epochs_list[0].ch_names)
    for ep in epochs_list[1:]:
        common &= set(ep.ch_names)
    ordered = [ch for ch in epochs_list[0].ch_names if ch in common]

    aligned = [ep.copy().pick(ordered) for ep in epochs_list]
    return mne.concatenate_epochs(aligned, verbose="ERROR")
