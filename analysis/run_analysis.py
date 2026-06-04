"""
TrueDetective CIT P300 — end-to-end analysis driver.

Run from the repo root:   python -m analysis.run_analysis

Pipeline per the approved design:
    load XDF → drop bad channel (A2) → filter/reref/ICA → epoch →
    pool sessions → P300 + bootstrap + cluster stats → behavioral RT/acc →
    write JSON/CSV to results/ and PNG figures to figures/.

Everything is driven by config.py; this module only orchestrates and reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")                 # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import mne
import numpy as np

from . import config as cfg
from .behavioral import analyze_behavioral
from .epoching import build_session_epochs, pool_epochs
from .erp_stats import analyze_p300, evokeds_by_condition
from .io_xdf import load_session
from .preprocess import preprocess


def _json_default(o):
    """Coerce numpy scalars/arrays so json.dumps can serialise the report."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


# ── Discovery ──────────────────────────────────────────────────────────
def find_xdf() -> list:
    return sorted(cfg.DATA_DIR.glob("*.xdf"))


def find_behavioral() -> list:
    return sorted(cfg.DATA_DIR.glob("behavioral_data_*.csv"))


# ── Figures ────────────────────────────────────────────────────────────
def plot_erps(evokeds: dict, channel: str, path) -> None:
    """Condition ERPs at the P300 site, plus the secret-irrelevant difference."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"secret": "C3", "irrelevant": "C0", "target": "C2"}
    for cond, ev in evokeds.items():
        if channel not in ev.ch_names:
            continue
        sig = ev.copy().pick([channel]).data[0] * cfg.MICROVOLT
        ax.plot(ev.times, sig, label=cond, color=colors.get(cond, "k"), lw=1.5)

    if "secret" in evokeds and "irrelevant" in evokeds:
        p = evokeds["secret"].copy().pick([channel]).data[0]
        i = evokeds["irrelevant"].copy().pick([channel]).data[0]
        ax.plot(evokeds["secret"].times, (p - i) * cfg.MICROVOLT,
                label="secret − irrelevant", color="k", ls="--", lw=1.2)

    ax.axvspan(*cfg.P300_WINDOW, color="grey", alpha=0.15, label="P300 window")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.invert_yaxis()                  # ERP convention: positive plotted down
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"CIT ERPs at {channel}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_topomap(evokeds: dict, path) -> None:
    """Scalp distribution of the secret−irrelevant difference in the P300 window."""
    if "secret" not in evokeds or "irrelevant" not in evokeds:
        return
    diff = mne.combine_evoked([evokeds["secret"], evokeds["irrelevant"]],
                              weights=[1, -1])
    if not diff.get_montage():
        return
    try:
        fig = diff.plot_topomap(times=[np.mean(cfg.P300_WINDOW)],
                                average=np.diff(cfg.P300_WINDOW)[0],
                                show=False)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        mne.utils.warn(f"topomap skipped: {exc}")


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    xdf_paths = find_xdf()
    if not xdf_paths:
        raise SystemExit(f"No XDF files found in {cfg.DATA_DIR}")

    print(f"Found {len(xdf_paths)} XDF session(s).")
    per_session_report = []
    epochs_list = []

    for path in xdf_paths:
        print(f"\n── {path.name} ──")
        session = load_session(path)
        pre = preprocess(session)

        if pre.bads:
            for ch, why in pre.bad_reasons.items():
                print(f"   bad channel {ch}: {why}")
        else:
            print("   no bad channels detected")
        print(f"   ICA excluded {pre.n_ica_excluded} component(s)")

        epochs = build_session_epochs(session, pre)
        epochs_list.append(epochs)
        counts = {c: int(len(epochs[c])) for c in cfg.EVENT_ID
                  if c in epochs.event_id}
        print(f"   epochs kept: {counts}")

        per_session_report.append({
            "session": session.name,
            "bad_channels": pre.bad_reasons,
            "ica_excluded": pre.ica_excluded,
            "epoch_counts": counts,
        })

    # ── Pool & analyse ─────────────────────────────────────────────────
    pooled = pool_epochs(epochs_list)
    print(f"\nPooled epochs: {len(pooled)} total")

    p300 = analyze_p300(pooled)
    print(f"\nP300 @ {p300.channel}:")
    print(f"   secret={p300.secret_uv:.2f}µV  irrelevant={p300.irrelevant_uv:.2f}µV"
          f"  diff={p300.diff_uv:+.2f}µV")
    print(f"   bootstrap p={p300.boot_p:.4f}  "
          f"95%CI=[{p300.boot_ci[0]:.2f}, {p300.boot_ci[1]:.2f}]µV")
    print(f"   CONCEALED INFORMATION {'DETECTED' if p300.detected else 'NOT detected'} "
          f"(α={cfg.DETECTION_ALPHA})")
    if p300.cluster_windows:
        for (lo, hi), pv in zip(p300.cluster_windows, p300.cluster_p):
            print(f"   sig. cluster {lo:.3f}–{hi:.3f}s (p={pv:.3f})")

    # ── Behavioral ─────────────────────────────────────────────────────
    behav = None
    csvs = find_behavioral()
    if csvs:
        behav = analyze_behavioral(csvs)
        print("\nBehavioral:")
        for cond, s in behav.by_condition.items():
            print(f"   {cond:11s} acc={s['accuracy']:.2f}  "
                  f"RT={s['rt_mean']*1000:.0f}±{s['rt_sd']*1000:.0f}ms (n={s['n']})")
        print(f"   secret vs irrelevant RT: t={behav.secret_vs_irr_rt_t:.2f}, "
              f"p={behav.secret_vs_irr_rt_p:.4f}")

    # ── Figures ────────────────────────────────────────────────────────
    evokeds = evokeds_by_condition(pooled)
    plot_erps(evokeds, p300.channel, cfg.FIGURES_DIR / "erp_p300.png")
    plot_topomap(evokeds, cfg.FIGURES_DIR / "topomap_secret_minus_irrelevant.png")
    print(f"\nFigures written to {cfg.FIGURES_DIR}")

    # ── Persist results ────────────────────────────────────────────────
    out = {
        "sessions": per_session_report,
        "pooled_epochs": int(len(pooled)),
        "p300": asdict(p300),
        "behavioral": asdict(behav) if behav else None,
        "config": {
            "reference": cfg.REFERENCE,
            "band": [cfg.HP_FREQ, cfg.LP_FREQ],
            "p300_window": list(cfg.P300_WINDOW),
            "detection_alpha": cfg.DETECTION_ALPHA,
        },
    }
    res_path = cfg.RESULTS_DIR / "analysis_results.json"
    res_path.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"Results written to {res_path}")


if __name__ == "__main__":
    main()
