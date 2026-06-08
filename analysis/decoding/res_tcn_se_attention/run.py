"""
Train & evaluate Res-TCN-SE-Attention on our CIT epochs (secret vs irrelevant).

Protocol (matched to the linear baseline for comparability):
  • Same pooled, cleaned epochs; targets excluded; ROC-AUC the headline metric.
  • Stratified 5-fold CV. Within each training fold a small validation split
    selects the best epoch (max val AUC). Out-of-fold test probabilities are
    pooled for an overall ROC / accuracy.
  • Class imbalance (~1:8) handled with a BCE pos_weight; raw input gets light
    Gaussian-noise augmentation; Adam + cosine-warm-restarts; the paper's
    150-epoch budget is trimmed to keep single-subject training honest.

Reality check: the paper's 99.94% came from 27 subjects with heavily
overlapping windows. With ~460 single-subject trials this model is
data-starved, so treat its AUC as an upper-effort deep baseline, not a ceiling.

Run:  python -m analysis.decoding.res_tcn_se_attention.run
"""

from __future__ import annotations

# Allow direct execution (python analysis/decoding/res_tcn_se_attention/run.py)
# as well as module execution (python -m analysis.decoding...run).
import pathlib as _pl
import sys as _sys
if __package__ in (None, ""):
    for _root in _pl.Path(__file__).resolve().parents:
        if (_root / "analysis" / "__init__.py").exists():
            _sys.path.insert(0, str(_root))
            break

import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from analysis import config as cfg
from analysis.decoding import common
from analysis.decoding.res_tcn_se_attention import features as featlib
from analysis.decoding.res_tcn_se_attention.model import ResTCNSEAttention

MODEL = "res_tcn_se_attention"


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(raw, dwt, feats, y, shuffle):
    ds = TensorDataset(
        torch.tensor(raw, dtype=torch.float32),
        torch.tensor(dwt, dtype=torch.float32),
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=cfg.TCN_BATCH, shuffle=shuffle)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    probs = []
    for raw, dwt, feats, _ in loader:
        logit = model(raw.to(device), dwt.to(device), feats.to(device))
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(probs)


def _train_fold(data, tr_idx, te_idx, device, record_history=False):
    """Train one CV fold; return (test_probs, best_val_auc, history)."""
    raw_all, dwt_all, feats_all, y = (data["raw"], data["dwt"],
                                      data["feats"], data["y"])

    # Inner train/val split for model selection (stratified).
    tr, va = train_test_split(
        tr_idx, test_size=cfg.TCN_VAL_FRAC, random_state=cfg.TCN_SEED,
        stratify=y[tr_idx])

    # Standardise the hand-crafted features on the *training* rows only.
    dwt_sc = StandardScaler().fit(dwt_all[tr])
    feats_sc = StandardScaler().fit(feats_all[tr])
    dwt = dwt_sc.transform(dwt_all).astype(np.float32)
    feats = feats_sc.transform(feats_all).astype(np.float32)

    tr_loader = _loader(raw_all[tr], dwt[tr], feats[tr], y[tr], shuffle=True)
    va_loader = _loader(raw_all[va], dwt[va], feats[va], y[va], shuffle=False)
    te_loader = _loader(raw_all[te_idx], dwt[te_idx], feats[te_idx],
                        y[te_idx], shuffle=False)

    model = ResTCNSEAttention(raw_all.shape[1], dwt_all.shape[1],
                              feats_all.shape[1]).to(device)
    n_pos = max(int((y[tr] == 1).sum()), 1)
    n_neg = int((y[tr] == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.TCN_LR,
                           weight_decay=cfg.TCN_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20)

    best_auc, best_state, history = -1.0, None, []
    for epoch in range(cfg.TCN_EPOCHS):
        model.train()
        ep_loss = 0.0
        for raw, dwt_b, feats_b, yb in tr_loader:
            opt.zero_grad()
            logit = model(raw.to(device), dwt_b.to(device), feats_b.to(device))
            loss = criterion(logit, yb.to(device))
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(yb)
        sched.step()

        va_prob = _predict(model, va_loader, device)
        va_auc = roc_auc_score(y[va], va_prob) if len(set(y[va])) > 1 else 0.5
        if record_history:
            history.append((ep_loss / len(tr), float(va_auc)))
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return _predict(model, te_loader, device), best_auc, history


# ── Figures ────────────────────────────────────────────────────────────
def plot_roc(y, prob, auc, path):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color="C3", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Res-TCN-SE-Attention — out-of-fold ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cv(fold_aucs, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(fold_aucs) + 1), fold_aucs, color="C0")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.axhline(np.mean(fold_aucs), color="C3", lw=1.5,
               label=f"mean {np.mean(fold_aucs):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Per-fold decoding AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path):
    if not history:
        return
    loss, auc = zip(*history)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(loss, color="C0", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(auc, color="C3", label="val AUC")
    ax2.axhline(0.5, color="k", ls="--", lw=0.8)
    ax2.set_ylabel("Val ROC-AUC", color="C3")
    ax1.set_title("Training curve (fold 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    torch.manual_seed(cfg.TCN_SEED)
    np.random.seed(cfg.TCN_SEED)
    fig_dir, res_dir = common.out_dirs(MODEL)
    device = _device()

    data = common.load_binary(verbose=True)
    X, y, ch_names = data["X"], data["y"], data["ch_names"]
    sfreq = data["epochs"].info["sfreq"]

    # Per-trial z-score (raw branch), then extract DWT + FEATS on the same data.
    Xz = common.zscore_per_trial(X).astype(np.float32)
    print(f"\nExtracting DWT + FEATS features for {len(Xz)} trials "
          f"({len(ch_names)} channels)…")
    dwt, feats = featlib.extract_all(Xz, sfreq, ch_names)
    print(f"   raw {Xz.shape}  dwt {dwt.shape}  feats {feats.shape}")

    data.update({"raw": Xz, "dwt": dwt, "feats": feats})
    print(f"Device: {device}  |  "
          f"{int((y == 1).sum())} secret vs {int((y == 0).sum())} irrelevant")

    oof = np.full(len(y), np.nan, dtype=float)
    fold_aucs, history0 = [], []
    cv = common.stratified_folds(y, n_splits=cfg.TCN_CV_FOLDS, seed=cfg.TCN_SEED)
    for k, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        probs, val_auc, hist = _train_fold(
            data, tr_idx, te_idx, device, record_history=(k == 0))
        oof[te_idx] = probs
        fa = roc_auc_score(y[te_idx], probs) if len(set(y[te_idx])) > 1 else 0.5
        fold_aucs.append(float(fa))
        if k == 0:
            history0 = hist
        print(f"   fold {k+1}: test AUC={fa:.3f}  (best val AUC={val_auc:.3f})")

    oof_auc = float(roc_auc_score(y, oof))
    # Youden-optimal threshold from the out-of-fold ROC.
    fpr, tpr, thr = roc_curve(y, oof)
    j_thr = float(thr[int(np.argmax(tpr - fpr))])
    pred = (oof >= j_thr).astype(int)
    bal_acc = float(balanced_accuracy_score(y, pred))

    n_params = ResTCNSEAttention(X.shape[1], dwt.shape[1], feats.shape[1]).n_params()

    print(f"\nRes-TCN-SE-Attention:")
    print(f"   per-fold AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    print(f"   out-of-fold AUC = {oof_auc:.3f}")
    print(f"   balanced accuracy @Youden = {bal_acc:.3f}")
    print(f"   trainable params = {n_params:,}")

    plot_roc(y, oof, oof_auc, fig_dir / "roc.png")
    plot_cv(fold_aucs, fig_dir / "cv_auc.png")
    plot_history(history0, fig_dir / "training.png")
    print(f"\nFigures written to {fig_dir}")

    out = {
        "model": MODEL,
        "classes": {cfg.DECODE_POSITIVE: 1, cfg.DECODE_NEGATIVE: 0},
        "n_secret": int((y == 1).sum()),
        "n_irrelevant": int((y == 0).sum()),
        "n_channels": int(X.shape[1]),
        "dwt_dim": int(dwt.shape[1]),
        "feats_dim": int(feats.shape[1]),
        "cv_folds": cfg.TCN_CV_FOLDS,
        "epochs": cfg.TCN_EPOCHS,
        "trainable_params": int(n_params),
        "fold_auc": fold_aucs,
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "oof_auc": oof_auc,
        "youden_threshold": j_thr,
        "balanced_accuracy": bal_acc,
        "device": str(device),
    }
    (res_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"Results written to {res_dir / 'results.json'}")


if __name__ == "__main__":
    main()
