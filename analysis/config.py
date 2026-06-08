"""
Central configuration for the TrueDetective CIT P300 analysis pipeline.

All tunable parameters live here so the processing scripts stay declarative.
Paths are resolved relative to the repository root, so the pipeline runs the
same regardless of the working directory.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── Stream / channel layout (CGX Quick-32r) ────────────────────────────
# The recording exposes 37 "channels"; only the first block are scalp EEG.
# Everything in NON_EEG is dropped (we time-lock with the LSL marker stream,
# not the hardware TRIGGER channel).
NON_EEG_CHANNELS = [
    "ExG 1", "ExG 2", "ACC32", "ACC33", "ACC34", "Packet Counter", "TRIGGER",
]
# Frontal channels used as EOG proxies for ICA blink detection.
EOG_PROXY_CHANNELS = ["Fp1", "Fp2"]

# ── Bad-channel detection ──────────────────────────────────────────────
# A2 (right mastoid) was disconnected in both sessions: impedance ~10k vs
# ~1k elsewhere, and the trace rails. These thresholds catch it generically.
IMPEDANCE_ABS_MAX = 5000.0        # absolute impedance ceiling
IMPEDANCE_REL_FACTOR = 5.0        # or > this * median(all channels)
VARIANCE_ROBUST_Z = 5.0           # robust z-score of per-channel std
FLAT_STD_FLOOR = 1e-7             # volts; below this a channel is "flat"

# ── Referencing & filtering ────────────────────────────────────────────
REFERENCE = "average"             # mastoid is dead -> common average reference
LINE_FREQ = 60.0                  # mains (US); set 50.0 in Europe
HP_FREQ = 0.1                     # high-pass (Hz)
LP_FREQ = 30.0                    # low-pass (Hz)

# ── ICA (ocular artifact removal) ──────────────────────────────────────
ICA_N_COMPONENTS = 0.95           # explain 95% variance
ICA_METHOD = "fastica"
ICA_RANDOM_STATE = 97
ICA_FIT_HP = 1.0                  # fit ICA on a 1 Hz high-passed copy

# ── Epoching ───────────────────────────────────────────────────────────
# Marker codes are unchanged (1/2/3); the "probe" condition is now labelled
# "secret" to match the experiment. Older CSVs that still say "probe" are
# normalised to "secret" in behavioral.py.
EVENT_ID = {"secret": 1, "irrelevant": 2, "target": 3}
TMIN, TMAX = -0.2, 0.8            # seconds around stimulus onset
BASELINE = (-0.2, 0.0)
REJECT_UV = 120.0                 # ±µV peak-to-peak epoch rejection
DECIM = 2                         # 500 Hz -> 250 Hz at epoching

# ── P300 measurement ───────────────────────────────────────────────────
P300_CHANNEL = "Pz"               # classic midline-parietal P300 site
P300_SECONDARY = ["Pz", "CPz", "Cz", "P3", "P4"]   # missing ones are ignored
P300_WINDOW = (0.30, 0.60)        # seconds; mean-amplitude window

# ── Statistics ─────────────────────────────────────────────────────────
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 2026
DETECTION_ALPHA = 0.05            # one-sided 95% for "secret > irrelevant"
CLUSTER_PERMUTATIONS = 1000
CLUSTER_SEED = 2026
CLUSTER_MIN_DURATION = 0.025      # s; ignore sub-component "clusters" (noise)

# ── Decoding (time-resolved MVPA) ──────────────────────────────────────
# Binary decoding of the concealed item vs neutral items. Targets are ignored
# entirely. ROC-AUC is the metric because the classes are imbalanced
# (~1 secret per ~8 irrelevants); chance AUC = 0.5 regardless of that ratio,
# and the classifier is class-weight balanced.
DECODE_POSITIVE = "secret"        # class 1
DECODE_NEGATIVE = "irrelevant"    # class 0  (targets excluded)
DECODE_CV_FOLDS = 5
DECODE_SCORING = "roc_auc"
DECODE_SEED = 2026
DECODE_CHANNEL_WINDOW = (0.0, 0.8)   # post-stim window for per-channel decoding
DECODE_TOP_K = 8                  # how many "most decodable" channels to flag

# ── Res-TCN-SE-Attention (deep decoder; Nicolescu et al. 2026) ─────────
# Faithful adaptation of the paper's best subject-dependent model to our
# 28-channel / 250 Hz CIT epochs. NB: that 99.94% was on 27 subjects with
# heavily-overlapping windows; on our single-subject ~460-trial set the model
# is expected to be power-limited and is reported with honest cross-validated
# AUC, alongside the linear baseline.
TCN_FILTERS = (64, 96, 128, 192, 256)   # per residual block
TCN_DILATIONS = (1, 2, 4, 8, 12)        # per residual block
TCN_INIT_KERNEL = 5
TCN_BLOCK_KERNEL = 3
TCN_RAW_EMBED = 192               # raw-branch dense embedding
TCN_DWT_EMBED = 128
TCN_FEATS_EMBED = 192             # first FEATS dense; second projects to 128
TCN_HEAD_UNITS = 160
TCN_DROPOUT = 0.3
TCN_SPATIAL_DROPOUT = 0.1
TCN_NOISE_STD = 0.01              # Gaussian input noise (raw branch, train only)
TCN_EPOCHS = 120
TCN_BATCH = 128
TCN_LR = 1e-3
TCN_WEIGHT_DECAY = 1e-4
TCN_VAL_FRAC = 0.15              # inner validation split for model selection
TCN_CV_FOLDS = 5
TCN_SEED = 2026
DWT_WAVELET = "db4"
DWT_LEVEL = 4

# ── Misc ───────────────────────────────────────────────────────────────
MONTAGE = "standard_1020"
MICROVOLT = 1e6                   # volts -> µV for reporting/plots
