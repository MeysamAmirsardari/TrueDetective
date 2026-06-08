"""Decoding models for TrueDetective (secret vs irrelevant).

Each model lives in its own sub-package with a runnable ``run`` module:

    analysis/decoding/
        common.py                       shared data prep / CV / metrics
        sliding_logreg/run.py           linear time-resolved MVPA baseline
        res_tcn_se_attention/run.py     Res-TCN-SE-Attention (Nicolescu et al.)

Run a model from the repo root, e.g.::

    python -m analysis.decoding.sliding_logreg.run
    python -m analysis.decoding.res_tcn_se_attention.run
"""
