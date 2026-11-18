"""
13_amputation_cuminc.py
------------------------
NEW 2026-11-10 (B8 in notes/2026-11-10_review2_sensitivity_prespec.md).
Weighted Aalen-Johansen cumulative incidence of major amputation at 1 year with death as a
competing event, per arm, from the PRIMARY amputation CCW datasets (combined weight, 48-h
grace). Risk difference ACEI/ARB minus comparator with a patient-level bootstrap CI (1,000
reps, seed 42; weights held fixed, as in 06.cox_hr_bootstrap).

A retained clone whose follow-up ends without amputation is classed as a competing death
when its end time equals the admission's death time (amputation follow-up is censored at
death in the primary data); otherwise it is censored.

Output: results/tables/amputation_cuminc.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from importlib import import_module
primary = import_module("06_primary_analysis")

H = 365


def aj_cif(time, cause, w, horizon=H):
    """Weighted Aalen-Johansen CIF for cause 1 at `horizon`; cause 0 = censored, 2 = competing."""
    order = np.argsort(time, kind="mergesort")
    t, c, w = np.asarray(time)[order], np.asarray(cause)[order], np.asarray(w, float)[order]
    at_risk = np.cumsum(w[::-1])[::-1]
    s, cif = 1.0, 0.0
    for u in np.unique(t[(c > 0) & (t <= horizon)]):
        i = np.searchsorted(t, u, side="left")
        n = at_risk[i]
        d1 = w[(t == u) & (c == 1)].sum()
        d2 = w[(t == u) & (c == 2)].sum()
        cif += s * d1 / n
        s *= 1 - (d1 + d2) / n
    return cif


def arm_cifs(df):
    r = {}
    for arm in (1, 0):
        m = (df.treated == arm).values
        r[arm] = aj_cif(df["time"].values[m], df["cause"].values[m], df["combined_weight"].values[m])
    return r[1], r[0]


if __name__ == "__main__":
    out = pd.read_parquet(data("outcomes.parquet")).set_index("hadm_id")
    rows, lines = [], []
    for comp in ["CCB", "betablocker", "diuretic"]:
        df, _ = primary.load_weighted(comp, "amputation")
        md = df["hadm_id"].map(out["mortality_days_from_t0"])
        me = df["hadm_id"].map(out["mortality_event"]).astype(bool)
        death = (df["event"] == 0) & me & (np.abs(df["time"] - md) < 1e-6)
        df["cause"] = np.where(df["event"] == 1, 1, np.where(death, 2, 0))
        ct, cc = arm_cifs(df)
        rng = np.random.default_rng(SEED)
        ids = df["subject_id"].unique()
        rds = []
        for b in range(N_BOOT):
            cnt = pd.Series(rng.choice(ids, size=len(ids), replace=True)).value_counts()
            rep = df.loc[df.index.repeat(df["subject_id"].map(cnt).fillna(0).astype(int))]
            a, c = arm_cifs(rep)
            rds.append(a - c)
        lo, hi = np.percentile(rds, [2.5, 97.5])
        rows.append({"comparison": comp, "cif_1y_treated": ct, "cif_1y_control": cc, "rd_1y": ct - cc,
                     "rd_1y_CI_lower": lo, "rd_1y_CI_upper": hi,
                     "competing_deaths_treated": int(((df.cause == 2) & (df.treated == 1)).sum()),
                     "competing_deaths_control": int(((df.cause == 2) & (df.treated == 0)).sum())})
        lines.append(f"{comp}: CIF ACEI/ARB {100*ct:.1f}% vs {100*cc:.1f}%, RD {100*(ct-cc):.1f} pp "
                     f"[{100*lo:.1f}, {100*hi:.1f}]")
        print(lines[-1], flush=True)
    pd.DataFrame(rows).to_csv(results_tables("amputation_cuminc.csv"), index=False)
    with open(log_path("13_amputation_cuminc.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
