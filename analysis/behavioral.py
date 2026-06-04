"""
Behavioral analysis from the experiment CSV logs.

The button-press data is a useful cross-check on the EEG result. In a CIT the
secret is something the participant is actively concealing, so even though they
press the same "deny" key as for irrelevants, the conflict often shows up as a
slower, less accurate response to the secret. We summarise reaction time and
accuracy per condition and run a quick secret-vs-irrelevant RT contrast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg


@dataclass
class BehavioralResult:
    n_trials: int
    by_condition: dict[str, dict[str, float]]      # cond -> {acc, rt_mean, ...}
    secret_vs_irr_rt_t: float
    secret_vs_irr_rt_p: float
    sources: list[str] = field(default_factory=list)


def load_behavioral(paths: list[Path]) -> pd.DataFrame:
    """Concatenate the per-session experiment CSVs into one tidy frame.

    Older recordings used the label "probe" (and a "session_probe" column)
    before the rename to "secret"; we normalise both so old and new CSVs pool
    cleanly under the current vocabulary.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source"] = Path(p).name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    out["trial_type"] = out["trial_type"].replace({"probe": "secret"})
    if "session_probe" in out.columns:
        out = out.rename(columns={"session_probe": "session_secret"})
    return out


def _condition_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Accuracy and correct-trial RT summary for each trial type."""
    out: dict[str, dict[str, float]] = {}
    for cond, g in df.groupby("trial_type"):
        correct = g[g["correct"] == 1]
        rts = correct["rt_seconds"].to_numpy(dtype=float)
        rts = rts[np.isfinite(rts)]
        out[str(cond)] = {
            "n": int(len(g)),
            "accuracy": float(g["correct"].mean()),
            "rt_mean": float(rts.mean()) if rts.size else float("nan"),
            "rt_median": float(np.median(rts)) if rts.size else float("nan"),
            "rt_sd": float(rts.std(ddof=1)) if rts.size > 1 else float("nan"),
        }
    return out


def analyze_behavioral(paths: list[Path]) -> BehavioralResult:
    """Summarise RT/accuracy and contrast secret vs irrelevant RT."""
    df = load_behavioral(paths)
    by_cond = _condition_stats(df)

    # Secret-vs-irrelevant RT (correct trials only): Welch's t-test, two-sided.
    correct = df[df["correct"] == 1]
    secret_rt = correct.loc[correct["trial_type"] == "secret", "rt_seconds"]
    irr_rt = correct.loc[correct["trial_type"] == "irrelevant", "rt_seconds"]
    secret_rt = secret_rt.to_numpy(dtype=float)
    irr_rt = irr_rt.to_numpy(dtype=float)
    secret_rt = secret_rt[np.isfinite(secret_rt)]
    irr_rt = irr_rt[np.isfinite(irr_rt)]

    if secret_rt.size > 1 and irr_rt.size > 1:
        t, p = stats.ttest_ind(secret_rt, irr_rt, equal_var=False)
    else:
        t, p = float("nan"), float("nan")

    return BehavioralResult(
        n_trials=int(len(df)),
        by_condition=by_cond,
        secret_vs_irr_rt_t=float(t),
        secret_vs_irr_rt_p=float(p),
        sources=sorted(df["source"].unique().tolist()),
    )
