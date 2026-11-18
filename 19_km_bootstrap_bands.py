"""
19_km_bootstrap_bands.py
------------------------
NEW 2026-11-13 (review v3, R3-m3; pre-specified in notes/2026-11-13_mi_bp_figures_length_prespec.md, F2).
Pointwise 95% confidence bands for the weighted Kaplan-Meier curves of Fig. 4 and Fig. S4: percentile
intervals from N_BOOT patient-level bootstrap replicates (SEED), with the weights fixed, as in the primary
bootstrap. A replicate is applied as patient multiplicities x weight, which is exactly equivalent to
repeating the resampled patients' rows.

Output: results/tables/km_bands.csv (comparison, outcome, arm, day, surv, lo, hi)
"""
import os
import sys
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from importlib import import_module
primary = import_module("06_primary_analysis")

GRID = np.arange(0, 366)


def wkm_matrix(t, e, W, grid):
    """Weighted KM for B weight vectors at once. t, e: (n,), W: (B, n). Returns (B, len(grid))."""
    o = np.argsort(t, kind="stable")
    t, e, W = t[o], e[o], W[:, o]
    ut, first = np.unique(t, return_index=True)
    at_risk_all = np.cumsum(W[:, ::-1], axis=1)[:, ::-1]           # sum of w with time >= t_i
    n_risk = at_risk_all[:, first]
    d = np.add.reduceat(W * e, first, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(n_risk > 0, d / n_risk, 0.0)
    S = np.cumprod(1 - h, axis=1)
    pos = np.searchsorted(ut, grid, side="right") - 1
    out = np.ones((W.shape[0], len(grid)))
    ok = pos >= 0
    out[:, ok] = S[:, pos[ok]]
    return out


if __name__ == "__main__":
    rows = []
    for comparison in ["CCB", "betablocker", "diuretic"]:
        for outcome in ["afs", "mortality", "amputation"]:
            df, _ = primary.load_weighted(comparison, outcome)
            pids = df["subject_id"].unique()
            pidx = pd.Series(np.arange(len(pids)), index=pids)[df["subject_id"]].values
            rng = np.random.default_rng(SEED)
            counts = np.zeros((N_BOOT, len(pids)))
            for b in range(N_BOOT):
                counts[b] = np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids))
            for arm, lab in [(1, "ACEI_ARB"), (0, comparison)]:
                m = (df.treated == arm).values
                t = df.loc[m, "time"].values.astype(float)
                e = df.loc[m, "event"].values.astype(float)
                w = df.loc[m, "combined_weight"].values.astype(float)
                point = wkm_matrix(t, e, w[None, :], GRID)[0]
                ref = KaplanMeierFitter().fit(t, e, weights=w).survival_function_at_times(GRID).values
                assert np.allclose(point, ref, atol=1e-9), (comparison, outcome, lab)
                boot = wkm_matrix(t, e, counts[:, pidx[m]] * w[None, :], GRID)
                lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
                rows.append(pd.DataFrame({"comparison": comparison, "outcome": outcome, "arm": lab, "day": GRID,
                                          "surv": point, "lo": lo, "hi": hi}))
                print(f"{comparison:12s} {outcome:10s} {lab:12s} S(365)={point[-1]:.3f} [{lo[-1]:.3f}, {hi[-1]:.3f}]")
    pd.concat(rows).to_csv(results_tables("km_bands.csv"), index=False)
