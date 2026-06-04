"""
XDF loading and conversion to MNE objects.

Each TrueDetective recording is an XDF file containing three LSL streams:
    • CGX Quick-32r EEG       (type "EEG",        500 Hz)
    • CGX Quick-32r Impedance (type "Impeadance", 500 Hz)  [sic, CGX spelling]
    • TrueDetective_Markers   (type "Markers",    irregular)

This module pulls the scalp EEG into an `mne.io.RawArray`, attaches a
10-20 montage, derives stimulus events from the marker stream (aligned on
the shared LSL clock), and returns the per-channel median impedance so the
preprocessing step can flag disconnected electrodes objectively.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pyxdf

from . import config as cfg


@dataclass
class SessionData:
    name: str
    raw: mne.io.BaseRaw
    events: np.ndarray                 # (n_events, 3): sample, 0, code
    impedance_med: dict[str, float]    # channel -> median impedance
    sfreq: float


def _find_stream(streams, type_substr):
    for s in streams:
        if type_substr.lower() in s["info"]["type"][0].lower():
            return s
    return None


def _channel_labels(stream):
    chans = stream["info"]["desc"][0]["channels"][0]["channel"]
    return [c["label"][0] for c in chans]


def load_session(path) -> SessionData:
    """Load one XDF file into a SessionData bundle."""
    streams, _ = pyxdf.load_xdf(
        str(path), synchronize_clocks=True, dejitter_timestamps=True
    )

    eeg = _find_stream(streams, "eeg")
    markers = _find_stream(streams, "marker")
    imp = _find_stream(streams, "mped") or _find_stream(streams, "mpead")
    if eeg is None or markers is None:
        raise RuntimeError(f"{path}: missing EEG or marker stream")

    labels = _channel_labels(eeg)
    sfreq = float(eeg["info"]["nominal_srate"][0])

    # ── Scalp EEG channels only ────────────────────────────────────────
    scalp_idx = [i for i, lab in enumerate(labels)
                 if lab not in cfg.NON_EEG_CHANNELS]
    scalp_labels = [labels[i] for i in scalp_idx]

    # pyxdf gives samples x channels, in microvolts -> volts for MNE.
    data = np.asarray(eeg["time_series"])[:, scalp_idx].T * 1e-6
    info = mne.create_info(scalp_labels, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    montage = mne.channels.make_standard_montage(cfg.MONTAGE)
    raw.set_montage(montage, on_missing="warn", match_case=False,
                    verbose="ERROR")

    # ── Median impedance per scalp channel (for bad-channel detection) ─
    impedance_med: dict[str, float] = {}
    if imp is not None:
        Z = np.asarray(imp["time_series"]).T          # ch x samples
        for i, lab in zip(scalp_idx, scalp_labels):
            impedance_med[lab] = float(np.median(Z[i]))

    # ── Marker stream -> MNE events on the EEG sample grid ─────────────
    eeg_ts = np.asarray(eeg["time_stamps"])
    mrk_ts = np.asarray(markers["time_stamps"])
    mrk_val = np.asarray(markers["time_series"]).astype(int).ravel()

    samples = np.searchsorted(eeg_ts, mrk_ts)
    samples = np.clip(samples, 0, len(eeg_ts) - 1)
    events = np.column_stack([
        samples, np.zeros_like(samples), mrk_val
    ]).astype(int)

    name = path.stem if hasattr(path, "stem") else str(path)
    return SessionData(
        name=name, raw=raw, events=events,
        impedance_med=impedance_med, sfreq=sfreq,
    )
