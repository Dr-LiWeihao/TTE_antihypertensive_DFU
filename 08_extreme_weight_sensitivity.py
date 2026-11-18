"""
08_extreme_weight_sensitivity.py
------------------------
NEW 2026-09-02. Addresses a reviewer-style concern (nature-reviewer pass,
R2-M2): eFigure 4 shows the IPCW component has a heavy right tail (99th
percentile combined weight up to ~44 even after the primary 1st/99th
percentile trim). This script checks whether the two Bonferroni-surviving
AFS point estimates (ACEI/ARB vs CCB, vs beta-blocker -- the primary
outcome's headline result) are stable under two stricter treatments of
the same extreme-weight tail, rather than leaving the concern as an
unquantified caveat in Discussion:

  (a) PRIMARY (for reference): combined weight trimmed at 1st/99th
      percentile (exactly reproduces 06's primary result).
  (b) STRICTER TRIM: combined weight trimmed at 2.5th/97.5th percentile.
  (c) EXCLUDE TOP 1%: admissions whose raw combined weight falls above
      the 99th percentile are dropped entirely (not capped), rather than
      capped at the 99th-percentile value -- a harder test than trimming.

UPDATED 2026-10-01 (after the IPCW/float32 fixes): runs all three AFS
comparisons, chosen independently of which ones are significant.
Originally re-ran only outcome="afs", comparison in {CCB, betablocker} -- the two
cells this sensitivity check was originally about (the diuretic/AFS and all
mortality/amputation-alone cells are not the ones eFigure 4's tail caveat
was raised against in Discussion). Reuses 06's exact propensity/weight/
Cox/bootstrap logic; the only change is the trim rule, isolated into
`build_combined_weight`.

Output: results/tables/extreme_weight_sensitivity.csv
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter

sys.path.insert(0, os.path.dirname(__file__))
from config import *

CONFOUNDERS_BASE = ["age_at_admission", "gender_male", "mi", "chf", "cerebrovascular",
                     "chronic_pulmonary", "renal", "sepsis", "charlson_comorbidity_index",
                     "baseline_creatinine", "creatinine_unknown", "baseline_sbp",
                     "baseline_dbp", "bp_unknown", "emergency_admission", "icu_stay_flag"]

BONFERRONI_ALPHA = 0.05 / 3
COMPARISONS = ["CCB", "betablocker", "diuretic"]  # all three AFS comparisons (2026-10-01: no longer
# restricted to the cells that were significant under the pre-fix results -- see docstring)
OUTCOME = "afs"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))


def e_value(estimate):
    rr = estimate if estimate >= 1 else 1 / estimate
    return rr + np.sqrt(rr * (rr - 1))


def build_combined_weight(df, rule):
    """rule: 'trim_1_99' (primary), 'trim_2p5_97p5' (stricter trim),
    or 'exclude_top1pct' (drop admissions above the 99th percentile
    of the raw combined weight, rather than capping them)."""
    raw = df["combined_weight_raw"]
    if rule == "trim_1_99":
        lo, hi = raw.quantile([0.01, 0.99])
        return df.assign(combined_weight=raw.clip(lower=lo, upper=hi)), None
    elif rule == "trim_2p5_97p5":
        lo, hi = raw.quantile([0.025, 0.975])
        return df.assign(combined_weight=raw.clip(lower=lo, upper=hi)), None
    elif rule == "exclude_top1pct":
        hi = raw.quantile(0.99)
        kept = df.loc[raw <= hi].copy()
        kept["combined_weight"] = kept["combined_weight_raw"]
        return kept, int((raw > hi).sum())
    else:
        raise ValueError(rule)


def run_one(comparison, rule):
    df = pd.read_parquet(data(f"ccw_{comparison}_{OUTCOME}.parquet"))
    df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
    confounders = [c for c in CONFOUNDERS_BASE if c in df.columns]
    race_cols = [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
    confounders += race_cols

    X = df[confounders].fillna(df[confounders].median()).values
    y = df["treated"].values
    ps_model = LogisticRegression(max_iter=2000)
    ps_model.fit(X, y, sample_weight=df["ipcw_weight"].values)
    df["ps"] = ps_model.predict_proba(X)[:, 1]
    marginal_p = np.average(df["treated"], weights=df["ipcw_weight"])
    df["iptw"] = np.where(df["treated"] == 1, marginal_p / df["ps"], (1 - marginal_p) / (1 - df["ps"]))
    df["combined_weight_raw"] = df["iptw"] * df["ipcw_weight"]

    analysis_df, n_dropped = build_combined_weight(df, rule)

    cox_df = analysis_df[["treated", "event", "time", "combined_weight", "subject_id"]].copy()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="time", event_col="event", weights_col="combined_weight",
            cluster_col="subject_id", robust=True, formula="treated")
    point_hr = cph.hazard_ratios_["treated"]
    cox_p = cph.summary.loc["treated", "p"]

    rng = np.random.default_rng(SEED)
    patient_ids = analysis_df["subject_id"].unique()
    boot_hrs = []
    for _ in range(N_BOOT):
        boot_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        counts = pd.Series(boot_ids).value_counts()
        row_mult = analysis_df["subject_id"].map(counts).fillna(0).astype(int)
        rep_df = analysis_df.loc[analysis_df.index.repeat(row_mult)].reset_index(drop=True)
        try:
            cph_b = CoxPHFitter()
            cph_b.fit(rep_df[["treated", "event", "time", "combined_weight"]],
                      duration_col="time", event_col="event", weights_col="combined_weight",
                      robust=True, formula="treated")
            boot_hrs.append(cph_b.hazard_ratios_["treated"])
        except Exception:
            pass
    boot_hrs = np.array(boot_hrs)
    ci_lo, ci_hi = np.percentile(boot_hrs, [2.5, 97.5])

    row = {
        "comparison": f"ACEI_ARB_vs_{comparison}", "outcome": OUTCOME, "weight_rule": rule,
        "n_analyzed": len(analysis_df), "n_dropped_top1pct": n_dropped,
        "HR": point_hr, "HR_CI_lower": ci_lo, "HR_CI_upper": ci_hi, "cox_p_value": cox_p,
        "passes_bonferroni_within_family": cox_p < BONFERRONI_ALPHA,
        "e_value_point": e_value(point_hr),
        "n_boot_successful": len(boot_hrs),
    }
    log(f"{comparison} / {rule}: N={len(analysis_df)} (dropped {n_dropped}), "
        f"HR={point_hr:.3f} [{ci_lo:.3f},{ci_hi:.3f}] p={cox_p:.4f}")
    return row


if __name__ == "__main__":
    rows = []
    for comparison in COMPARISONS:
        for rule in ["trim_1_99", "trim_2p5_97p5", "exclude_top1pct"]:
            rows.append(run_one(comparison, rule))
    out = pd.DataFrame(rows)
    out.to_csv(results_tables("extreme_weight_sensitivity.csv"), index=False)
    log("\nSaved: results/tables/extreme_weight_sensitivity.csv")
    with open(log_path("08_extreme_weight_sensitivity.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
