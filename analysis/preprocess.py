"""
Preprocessing: bad-channel handling, referencing, filtering, ICA.

The pipeline is deliberately objective about the dead electrode. A2 (right
mastoid) was disconnected during recording, so rather than hard-coding its
name we detect it from impedance + signal statistics. That keeps the code
honest if a *different* channel drops out in a future session.

Order of operations (matters):
    1.  Detect & drop bad channels (impedance, variance, flatness).
    2.  Notch out mains, then band-pass 0.1-30 Hz.
    3.  Common-average reference (the dead mastoid can't be a reference).
    4.  ICA on a 1 Hz high-passed copy; zero out ocular components,
        identified from the frontal (Fp1/Fp2) EOG proxies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np

from . import config as cfg
from .io_xdf import SessionData


@dataclass
class PreprocResult:
    raw: mne.io.BaseRaw
    bads: list[str] = field(default_factory=list)
    bad_reasons: dict[str, str] = field(default_factory=dict)
    n_ica_excluded: int = 0
    ica_excluded: list[int] = field(default_factory=list)


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score; robust to the very outliers we're hunting."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def detect_bad_channels(session: SessionData) -> dict[str, str]:
    """Return {channel: reason} for channels failing any quality check.

    Three independent criteria, any of which condemns a channel:
      • impedance  — absolute ceiling, or >> the array median
      • variance   — robust z-score of per-channel std is an outlier
                     (a railed/disconnected lead has enormous variance)
      • flatness   — std below the floor (a truly dead/shorted lead)
    """
    raw = session.raw
    data = raw.get_data()                       # ch x samples, volts
    stds = data.std(axis=1)
    reasons: dict[str, str] = {}

    # ── Impedance-based ────────────────────────────────────────────────
    imp = session.impedance_med
    if imp:
        imp_vals = np.array([imp.get(ch, np.nan) for ch in raw.ch_names])
        med_imp = np.nanmedian(imp_vals)
        for ch, z in zip(raw.ch_names, imp_vals):
            if np.isnan(z):
                continue
            if z > cfg.IMPEDANCE_ABS_MAX:
                reasons[ch] = f"impedance {z:.0f}Ω > {cfg.IMPEDANCE_ABS_MAX:.0f}Ω"
            elif med_imp > 0 and z > cfg.IMPEDANCE_REL_FACTOR * med_imp:
                reasons[ch] = (f"impedance {z:.0f}Ω > "
                               f"{cfg.IMPEDANCE_REL_FACTOR:.0f}× median "
                               f"({med_imp:.0f}Ω)")

    # ── Variance outliers ──────────────────────────────────────────────
    rz = _robust_z(stds)
    for ch, z, s in zip(raw.ch_names, rz, stds):
        if ch in reasons:
            continue
        if abs(z) > cfg.VARIANCE_ROBUST_Z:
            reasons[ch] = f"variance robust-z {z:.1f} (std {s*cfg.MICROVOLT:.1f}µV)"
        elif s < cfg.FLAT_STD_FLOOR:
            reasons[ch] = f"flat (std {s*cfg.MICROVOLT:.3f}µV)"

    return reasons


def preprocess(session: SessionData) -> PreprocResult:
    """Full single-session preprocessing → cleaned, referenced RawArray."""
    raw = session.raw.copy()

    # ── 1. Bad channels ────────────────────────────────────────────────
    bad_reasons = detect_bad_channels(session)
    bads = list(bad_reasons)
    raw.info["bads"] = bads
    if bads:
        # The leads are genuinely disconnected — interpolation would invent
        # data, so we drop them and rely on the average reference instead.
        raw.drop_channels(bads)

    # ── 2. Filtering ───────────────────────────────────────────────────
    # Notch first so the mains peak doesn't leak through the band-pass edges.
    raw.notch_filter(cfg.LINE_FREQ, verbose="ERROR")
    raw.filter(cfg.HP_FREQ, cfg.LP_FREQ, fir_design="firwin", verbose="ERROR")

    # ── 3. ICA ocular-artifact removal ─────────────────────────────────
    # Fit ICA *before* re-referencing: the common-average reference makes the
    # data rank-deficient (one degree of freedom removed), which would make the
    # PCA whitening invert a near-zero eigenvalue (divide-by-zero/overflow).
    ica_excluded: list[int] = []
    try:
        ica_excluded = _run_ica(raw)
    except Exception as exc:                     # ICA is best-effort
        mne.utils.warn(f"ICA skipped for {session.name}: {exc}")

    # ── 4. Common-average reference (after ICA, on the cleaned signal) ──
    raw.set_eeg_reference(cfg.REFERENCE, verbose="ERROR")

    return PreprocResult(
        raw=raw,
        bads=bads,
        bad_reasons=bad_reasons,
        n_ica_excluded=len(ica_excluded),
        ica_excluded=ica_excluded,
    )


def _run_ica(raw: mne.io.BaseRaw) -> list[int]:
    """Fit ICA on a 1 Hz high-passed copy, drop ocular components in-place.

    ICA decomposition is unstable on slow drifts, so we fit on an aggressively
    high-passed copy (standard MNE practice) and apply the unmixing to the
    real, 0.1 Hz data. Blink components are found by correlating with the
    frontal channels acting as EOG proxies.

    The whole computation runs under ``np.errstate(all="ignore")``: on newer
    numpy/scipy, ``scipy.linalg.pinv`` emits spurious "divide by zero / overflow
    / invalid in matmul" warnings while reconstructing its SVD, even though the
    input is full-rank and the result is finite and valid (verified downstream).
    Silencing them here keeps the run output clean without masking real issues
    elsewhere in the pipeline.
    """
    with np.errstate(all="ignore"):
        ica_raw = raw.copy().filter(cfg.ICA_FIT_HP, None,
                                    fir_design="firwin", verbose="ERROR")
        ica = mne.preprocessing.ICA(
            n_components=cfg.ICA_N_COMPONENTS,
            method=cfg.ICA_METHOD,
            random_state=cfg.ICA_RANDOM_STATE,
            max_iter="auto",
        )
        ica.fit(ica_raw, verbose="ERROR")

        # Use whichever frontal proxies survived bad-channel dropping.
        eog_chs = [ch for ch in cfg.EOG_PROXY_CHANNELS if ch in raw.ch_names]
        exclude: list[int] = []
        for ch in eog_chs:
            idx, _ = ica.find_bads_eog(raw, ch_name=ch, verbose="ERROR")
            exclude.extend(idx)

        ica.exclude = sorted(set(exclude))
        ica.apply(raw, verbose="ERROR")
    return ica.exclude
