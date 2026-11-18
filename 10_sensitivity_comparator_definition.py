"""
10_sensitivity_comparator_definition.py
----------------------------------------
NEW 2026-11-10 (pre-submission review; notes/2026-11-10_druglist_amputation_rerun.md).
Pre-specified BEFORE the re-run results were seen, in response to the concern that two
primary comparators mix antihypertensive and non-antihypertensive indications:
  - CCB arm included diltiazem/verapamil (often rate control);
  - diuretic arm was mostly loop diuretics, often IV (decongestion in heart failure).

Restricted comparator strategies (ACEI/ARB strategy unchanged):
  CCB_DHP            dihydropyridine CCB present, no ACEI/ARB (admissions on a
                     non-dihydropyridine CCB only -> "neither", censored at the grace end)
  diuretic_thiazide  thiazide-type diuretic present, NO loop diuretic, no ACEI/ARB

Same CCW / IPCW / IPTW / Cox / patient-bootstrap machinery as the primary analysis
(05.build_ccw_dataset, 06.add_combined_weight, 06.cox_hr_bootstrap), 48-h grace period,
all three outcomes.

Output: results/tables/sensitivity_comparator_definition.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from importlib import import_module
ccw_mod = import_module("05_ccw_clone_censor")
primary = import_module("06_primary_analysis")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))


def run(comparison, outcome):
    df = ccw_mod.build_ccw_dataset(GRACE_HOURS_PRIMARY, comparison, outcome, log=log)
    df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
    confounders = [c for c in primary.CONFOUNDERS_BASE if c in df.columns]
    confounders += [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
    df = primary.add_combined_weight(df, confounders)
    t = df.treated == 1
    n_imb = sum(abs(primary.smd(df.loc[t, c].values, df.loc[~t, c].values,
                                df.loc[t, "combined_weight"].values, df.loc[~t, "combined_weight"].values)) >= 0.10
                for c in confounders)
    fit = primary.cox_hr_bootstrap(df)
    row = {"comparison": f"ACEI_ARB_vs_{comparison}", "outcome": outcome, "grace_hours": GRACE_HOURS_PRIMARY,
           "n_treated": int(t.sum()), "n_control": int((~t).sum()),
           "events_treated": int(df.loc[t, "event"].sum()), "events_control": int(df.loc[~t, "event"].sum()),
           "HR": fit["HR"], "HR_CI_lower": fit["ci_lo"], "HR_CI_upper": fit["ci_hi"], "cox_p_value": fit["cox_p"],
           "rd_1y": fit["rd_1y"], "rd_1y_CI_lower": fit["rd_ci_lo"], "rd_1y_CI_upper": fit["rd_ci_hi"],
           "n_covariates_imbalanced": int(n_imb), "n_covariates": len(confounders),
           "n_boot_successful": fit["n_boot"]}
    log(f"  -> HR={fit['HR']:.3f} [{fit['ci_lo']:.3f}, {fit['ci_hi']:.3f}], p={fit['cox_p']:.3f}, "
        f"imbalanced {n_imb}/{len(confounders)}, events {row['events_treated']}/{row['events_control']}")
    return row


if __name__ == "__main__":
    rows = []
    for comparison in ccw_mod.SENSITIVITY_COMPARATORS:
        for outcome in ccw_mod.OUTCOMES:
            log(f"\n{'=' * 70}\n{comparison} / {outcome}\n{'=' * 70}")
            rows.append(run(comparison, outcome))
    out = pd.DataFrame(rows)
    out.to_csv(results_tables("sensitivity_comparator_definition.csv"), index=False)
    log("\n" + out.to_string())
    with open(log_path("10_sensitivity_comparator_definition.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
