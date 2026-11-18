"""
07_sensitivity_grace_period.py
---------------------------------
NEW 2026-08-31, EXTENDED same day to all three outcome families (AFS,
mortality, amputation-alone) per user request -- previously mortality
only. Grace-period sensitivity analysis (CLAUDE.md finalized statistical
analysis plan): reruns the CCW pipeline at GRACE_HOURS_SENSITIVITY (24h,
72h) alongside the PRIMARY 48h result, for all 3 outcomes x 3 comparisons
(27 combinations total).

Reuses 05's build_ccw_dataset() and 06's weighting/Cox logic (imported,
not duplicated) so the grace-period parameter is the only thing that
changes between primary and sensitivity runs.

Output: results/tables/sensitivity_grace_period.csv
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from importlib import import_module
ccw_mod = import_module("05_ccw_clone_censor")
primary_mod = import_module("06_primary_analysis")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("Grace-period sensitivity analysis -- all 3 outcomes x 3 comparisons")
log(f"Primary = {GRACE_HOURS_PRIMARY}h; sensitivity = {GRACE_HOURS_SENSITIVITY}")
log("=" * 70)

CONFOUNDERS_BASE = primary_mod.CONFOUNDERS_BASE
OUTCOME_NAMES = list(ccw_mod.OUTCOMES.keys())  # afs, mortality, amputation

results = []
for grace_hours in [GRACE_HOURS_PRIMARY] + GRACE_HOURS_SENSITIVITY:
    for comparison in ccw_mod.COMPARATORS:
      for outcome_name in OUTCOME_NAMES:
        df = ccw_mod.build_ccw_dataset(grace_hours, comparison, outcome_name, log=log)
        df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
        confounders = [c for c in CONFOUNDERS_BASE if c in df.columns]
        race_cols = [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
        confounders += race_cols

        # Combined-weight construction matches the fix in 06 (2026-08-31):
        # propensity fit WEIGHTED BY ipcw_weight, single trim on the
        # product -- see 06's inline comment for the diagnostic finding.
        X = df[confounders].fillna(df[confounders].median()).values
        y = df["treated"].values
        ps_model = LogisticRegression(max_iter=2000)
        ps_model.fit(X, y, sample_weight=df["ipcw_weight"].values)
        df["ps"] = ps_model.predict_proba(X)[:, 1]
        marginal_p = np.average(df["treated"], weights=df["ipcw_weight"])
        df["iptw"] = np.where(df["treated"] == 1, marginal_p / df["ps"], (1 - marginal_p) / (1 - df["ps"]))
        df["combined_weight_raw"] = df["iptw"] * df["ipcw_weight"]
        lo_c, hi_c = df["combined_weight_raw"].quantile([0.01, 0.99])
        df["combined_weight"] = df["combined_weight_raw"].clip(lower=lo_c, upper=hi_c)

        cox_df = df[["treated", "event", "time", "combined_weight", "subject_id"]].copy()
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time", event_col="event", weights_col="combined_weight",
                cluster_col="subject_id", robust=True, formula="treated")
        point_hr = cph.hazard_ratios_["treated"]
        cox_p = cph.summary.loc["treated", "p"]

        rng = np.random.default_rng(SEED)
        patient_ids = df["subject_id"].unique()
        boot_hrs = []
        for b in range(N_BOOT):
            boot_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            counts = pd.Series(boot_ids).value_counts()
            row_mult = df["subject_id"].map(counts).fillna(0).astype(int)
            rep_df = df.loc[df.index.repeat(row_mult)].reset_index(drop=True)
            try:
                cph_b = CoxPHFitter()
                cph_b.fit(rep_df[["treated", "event", "time", "combined_weight"]],
                          duration_col="time", event_col="event", weights_col="combined_weight",
                          robust=True, formula="treated")
                boot_hrs.append(cph_b.hazard_ratios_["treated"])
            except Exception:
                pass
        boot_hrs = np.array(boot_hrs)
        ci_lo, ci_hi = np.percentile(boot_hrs, [2.5, 97.5]) if len(boot_hrs) > 0 else (np.nan, np.nan)

        log(f"[grace={grace_hours}h] {comparison}/{outcome_name}: HR={point_hr:.3f} [{ci_lo:.3f},{ci_hi:.3f}] "
            f"p={cox_p:.3f} (boot n={len(boot_hrs)}/{N_BOOT})")
        results.append({"grace_hours": grace_hours, "comparison": comparison, "outcome": outcome_name,
                         "is_primary": grace_hours == GRACE_HOURS_PRIMARY,
                         "HR": point_hr, "HR_CI_lower": ci_lo, "HR_CI_upper": ci_hi, "cox_p_value": cox_p,
                         "n_boot_successful": len(boot_hrs)})

sens_df = pd.DataFrame(results)
sens_df.to_csv(results_tables("sensitivity_grace_period.csv"), index=False)
log(f"\nSaved: results/tables/sensitivity_grace_period.csv")
log(sens_df.to_string())

with open(log_path("07_sensitivity_grace_period.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
