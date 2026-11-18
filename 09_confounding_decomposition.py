"""
09_confounding_decomposition.py
--------------------------------
NEW 2026-10-01. EXPLORATORY, POST HOC (added after the corrected primary
analysis was null; labelled as such in the manuscript). Asks which baseline
factors explain the crude ACEI/ARB advantage, by re-estimating each
comparison with nested covariate sets used in BOTH the censoring (IPCW)
and treatment (IPTW) models:

  M0 crude          -- unweighted Cox on retained clones
  M1 demographics   -- age, sex, race
  M2a + cardiorenal -- M1 + renal disease, baseline creatinine (+ missing
                       indicator), congestive heart failure, myocardial
                       infarction
  M2b + other       -- M1 + all non-cardiorenal factors (cerebrovascular,
                       chronic pulmonary, sepsis, blood pressure (+ missing),
                       emergency admission, ICU stay); Charlson index
                       deliberately excluded because it contains renal
                       disease and heart failure
  M3 full           -- primary-analysis covariate set (identical to 06)

Weighting, trimming, Cox and bootstrap reuse 05.build_ccw_dataset and
06.add_combined_weight / cox_hr_bootstrap -- no re-implementation.
Primary outcome (AFS): 1000-rep patient bootstrap CIs (as in 06).
Secondary outcomes: robust (patient-clustered) Wald CIs only, to bound
run time; labelled as such.

Output: results/tables/confounding_decomposition.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from importlib import import_module
ccw_mod = import_module("05_ccw_clone_censor")
primary = import_module("06_primary_analysis")

DEMO = ["age_at_admission", "gender_male"]  # + race dummies
CARDIORENAL = ["renal", "baseline_creatinine", "creatinine_unknown", "chf", "mi"]
OTHER = ["cerebrovascular", "chronic_pulmonary", "sepsis", "baseline_sbp", "baseline_dbp", "bp_unknown",
         "emergency_admission", "icu_stay_flag"]
MODELS = {
    "M0_crude": None,
    "M1_demographics": DEMO,
    "M2a_demographics_cardiorenal": DEMO + CARDIORENAL,
    "M2b_demographics_other": DEMO + OTHER,
    "M3_full": "FULL",
}

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))


def robust_fit(df, weight_col):
    cols = ["treated", "event", "time", "subject_id"] + ([weight_col] if weight_col else [])
    cph = CoxPHFitter().fit(df[cols], duration_col="time", event_col="event", weights_col=weight_col,
                            cluster_col="subject_id", robust=True, formula="treated")
    ci = np.exp(cph.confidence_intervals_.loc["treated"].values)
    return {"HR": cph.hazard_ratios_["treated"], "cox_p": cph.summary.loc["treated", "p"],
            "ci_lo": ci[0], "ci_hi": ci[1], "n_boot": 0, "n_fail": 0}


def run(comparison, outcome, model, covs):
    if covs == "FULL":
        df, _ = primary.load_weighted(comparison, outcome)
        weight_col = "combined_weight"
    else:
        quiet = lambda *_: None
        if covs is None:
            df = ccw_mod.build_ccw_dataset(GRACE_HOURS_PRIMARY, comparison, outcome, log=quiet)
            weight_col = None
        else:
            df = ccw_mod.build_ccw_dataset(GRACE_HOURS_PRIMARY, comparison, outcome, log=quiet,
                                           ipcw_covariates=list(covs), include_race=True)
            weight_col = "combined_weight"
        df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
        df = df.reset_index(drop=True)
        if covs is not None:
            race_cols = [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
            df = primary.add_combined_weight(df, list(covs) + race_cols)
    fit = primary.cox_hr_bootstrap(df, weight_col) if outcome == "afs" else robust_fit(df, weight_col)
    ci_method = "bootstrap" if outcome == "afs" else "robust"
    log(f"{comparison:12s} {outcome:10s} {model:30s} HR={fit['HR']:.3f} [{fit['ci_lo']:.3f},{fit['ci_hi']:.3f}] "
        f"p={fit['cox_p']:.3f} ({ci_method})")
    return {"comparison": comparison, "outcome": outcome, "model": model, "ci_method": ci_method,
            "n_treated": int(df.treated.sum()), "n_control": int((1 - df.treated).sum()),
            "HR": fit["HR"], "HR_CI_lower": fit["ci_lo"], "HR_CI_upper": fit["ci_hi"], "cox_p_value": fit["cox_p"],
            "n_boot_successful": fit["n_boot"]}


if __name__ == "__main__":
    rows = []
    for outcome in ["afs", "mortality", "amputation"]:
        for comparison in ccw_mod.COMPARATORS:
            for model, covs in MODELS.items():
                rows.append(run(comparison, outcome, model, covs))
    out = pd.DataFrame(rows)
    out.to_csv(results_tables("confounding_decomposition.csv"), index=False)
    log("Saved: results/tables/confounding_decomposition.csv")
    with open(log_path("09_confounding_decomposition.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
