"""
15_betablocker_pad_exploration.py
----------------------------------
NEW 2026-11-10. EXPLORATORY, post hoc. Pre-specified before running in
notes/2026-11-10_betablocker_pad_exploration_prespec.md. Does not change the primary analysis.

Usage:
  python 15_betablocker_pad_exploration.py <task> [<task> ...]
  python 15_betablocker_pad_exploration.py combine        -> results/tables/bb_pad_exploration.csv

Tasks:
  E1_betablocker, E1_CCB, E1_diuretic   effect modification by CLTI (3 outcomes each)
  E1b_athero                            effect modification by atherosclerotic PAD (beta-blocker)
  E2_subtype                            ACEI/ARB vs vasodilating / non-vasodilating beta-blocker
  E3_anyamp                             any (major or minor) amputation: vs beta-blocker overall + by CLTI,
                                        vs CCB and diuretic overall

Same CCW / IPCW / IPTW / weighted Cox machinery as the primary analysis (48-h grace). Within
a stratum, IPCW and IPTW are refitted on that stratum. Stratum HRs, their CIs and the ratio of
HRs come from ONE joint patient-level bootstrap (1,000 reps, seed 42; weights held fixed, as in
06.cox_hr_bootstrap). The interaction p is a Wald test from a weighted Cox model on the stacked
strata (treated + stratum + treated:stratum), robust SE clustered on subject_id.
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

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

G = GRACE_HOURS_PRIMARY
OUTS = list(ccw_mod.OUTCOMES)  # afs, mortality, amputation
ext = pd.read_parquet(data("pad_bb_anyamp.parquet"))
FLAGS = ext[["hadm_id", f"bbvasoonly_{G}h", f"bbnonvasoonly_{G}h"]]
EXTRA = ext[["hadm_id", "clti", "athero_pad", "anyamp_at_t0"]]
ANYAMP = ext[["hadm_id"]].assign(amputation_only_event=ext["anyamp_event"],
                                 amputation_only_days_from_t0=ext["anyamp_days_from_t0"])
NO_DAY0_ANYAMP = lambda b: ~b["anyamp_at_t0"].astype(bool)


def build(comparison, outcome, admission_filter=None, outcome_override=None):
    df = ccw_mod.build_ccw_dataset(G, comparison, outcome, log=log, extra_covariates=EXTRA,
                                   admission_filter=admission_filter, outcome_override=outcome_override,
                                   extra_drug_flags=FLAGS)
    df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
    conf = [c for c in primary.CONFOUNDERS_BASE if c in df.columns]
    conf += [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
    attrs = dict(df.attrs)
    df = primary.add_combined_weight(df, conf)
    df.attrs.update(attrs)
    t = df.treated == 1
    n_imb = sum(abs(primary.smd(df.loc[t, c].values, df.loc[~t, c].values, df.loc[t, "combined_weight"].values,
                                df.loc[~t, "combined_weight"].values)) >= 0.10 for c in conf)
    return df, int(n_imb), len(conf)


def describe(df, n_imb, n_conf):
    t = df.treated == 1
    return {"n_admissions": df.attrs["n_admissions"], "n_treated": int(t.sum()), "n_control": int((~t).sum()),
            "events_treated": int(df.loc[t, "event"].sum()), "events_control": int(df.loc[~t, "event"].sum()),
            "n_covariates_imbalanced": n_imb, "n_covariates": n_conf}


def hr_point(df):
    cph = CoxPHFitter().fit(df[["treated", "event", "time", "combined_weight"]], "time", "event",
                            weights_col="combined_weight", robust=True)
    return cph.hazard_ratios_["treated"]


def robust_fit(df):
    cph = CoxPHFitter().fit(df[["treated", "event", "time", "combined_weight", "subject_id"]], "time", "event",
                            weights_col="combined_weight", cluster_col="subject_id", robust=True)
    return cph.hazard_ratios_["treated"], cph.summary.loc["treated", "p"]


def stratified(comparison, outcome, modifier, base_filter=None, outcome_override=None, task=""):
    """HR in modifier==1 and modifier==0 strata + ratio (1 / 0) from a joint patient bootstrap."""
    dfs, rows = {}, []
    for lev in (1, 0):
        f = (lambda b, lev=lev: (b[modifier] == lev) & (base_filter(b) if base_filter else True))
        log(f"\n=== {task} / {comparison} / {outcome} / {modifier}={lev} ===")
        df, n_imb, n_conf = build(comparison, outcome, f, outcome_override)
        dfs[lev] = (df, n_imb, n_conf)
    a, b = dfs[1][0], dfs[0][0]
    hr_a, p_a = robust_fit(a)
    hr_b, p_b = robust_fit(b)
    st = pd.concat([a.assign(stratum=1), b.assign(stratum=0)], ignore_index=True)
    st["treated_x_stratum"] = st["treated"] * st["stratum"]
    cph = CoxPHFitter().fit(st[["treated", "stratum", "treated_x_stratum", "event", "time", "combined_weight",
                                "subject_id"]], "time", "event", weights_col="combined_weight",
                            cluster_col="subject_id", robust=True)
    p_int = cph.summary.loc["treated_x_stratum", "p"]

    rng = np.random.default_rng(SEED)
    ids = pd.unique(np.concatenate([a["subject_id"].unique(), b["subject_id"].unique()]))
    ba, bb, br, fails = [], [], [], 0
    for _ in range(N_BOOT):
        cnt = pd.Series(rng.choice(ids, size=len(ids), replace=True)).value_counts()
        try:
            ra = a.loc[a.index.repeat(a["subject_id"].map(cnt).fillna(0).astype(int))]
            rb = b.loc[b.index.repeat(b["subject_id"].map(cnt).fillna(0).astype(int))]
            x, y = hr_point(ra), hr_point(rb)
            ba.append(x); bb.append(y); br.append(x / y)
        except Exception:
            fails += 1
    for lev, hr, p, boots in [(1, hr_a, p_a, ba), (0, hr_b, p_b, bb)]:
        df, n_imb, n_conf = dfs[lev]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"task": task, "comparison": comparison, "outcome": outcome, "subset": f"{modifier}={lev}",
                     **describe(df, n_imb, n_conf), "HR": hr, "HR_CI_lower": lo, "HR_CI_upper": hi, "cox_p_value": p,
                     "ratio_of_HR": np.nan, "ratio_CI_lower": np.nan, "ratio_CI_upper": np.nan,
                     "p_interaction": np.nan, "n_boot_successful": len(boots)})
    lo, hi = np.percentile(br, [2.5, 97.5])
    rows.append({"task": task, "comparison": comparison, "outcome": outcome,
                 "subset": f"ratio {modifier}=1 / {modifier}=0", "HR": np.nan,
                 "ratio_of_HR": hr_a / hr_b, "ratio_CI_lower": lo, "ratio_CI_upper": hi,
                 "p_interaction": p_int, "n_boot_successful": len(br)})
    log(f"  -> {modifier}=1 HR {hr_a:.3f} [{rows[0]['HR_CI_lower']:.3f}, {rows[0]['HR_CI_upper']:.3f}]; "
        f"{modifier}=0 HR {hr_b:.3f} [{rows[1]['HR_CI_lower']:.3f}, {rows[1]['HR_CI_upper']:.3f}]; "
        f"ratio {hr_a/hr_b:.3f} [{lo:.3f}, {hi:.3f}], p_int={p_int:.3f}; boot fails {fails}")
    return rows


def overall(comparison, outcome, admission_filter=None, outcome_override=None, task=""):
    log(f"\n=== {task} / {comparison} / {outcome} / overall ===")
    df, n_imb, n_conf = build(comparison, outcome, admission_filter, outcome_override)
    fit = primary.cox_hr_bootstrap(df)
    log(f"  -> HR {fit['HR']:.3f} [{fit['ci_lo']:.3f}, {fit['ci_hi']:.3f}] p={fit['cox_p']:.3f}; "
        f"RD {100*fit['rd_1y']:.1f} pp [{100*fit['rd_ci_lo']:.1f}, {100*fit['rd_ci_hi']:.1f}]")
    return [{"task": task, "comparison": comparison, "outcome": outcome, "subset": "overall",
             **describe(df, n_imb, n_conf), "HR": fit["HR"], "HR_CI_lower": fit["ci_lo"],
             "HR_CI_upper": fit["ci_hi"], "cox_p_value": fit["cox_p"], "rd_1y": fit["rd_1y"],
             "rd_1y_CI_lower": fit["rd_ci_lo"], "rd_1y_CI_upper": fit["rd_ci_hi"],
             "n_boot_successful": fit["n_boot"]}]


def run(task):
    rows = []
    if task.startswith("E1_"):
        comp = task[3:]
        for o in OUTS:
            rows += stratified(comp, o, "clti", task=task)
    elif task == "E1b_athero":
        for o in OUTS:
            rows += stratified("betablocker", o, "athero_pad", task=task)
    elif task == "E2_subtype":
        for comp in ccw_mod.EXPLORATORY_COMPARATORS:
            for o in OUTS:
                rows += overall(comp, o, task=task)
    elif task == "E3_anyamp":
        rows += overall("betablocker", "amputation", NO_DAY0_ANYAMP, ANYAMP, task)
        rows += stratified("betablocker", "amputation", "clti", NO_DAY0_ANYAMP, ANYAMP, task)
        for comp in ["CCB", "diuretic"]:
            rows += overall(comp, "amputation", NO_DAY0_ANYAMP, ANYAMP, task)
        for r in rows:
            r["outcome"] = "any_amputation"
    else:
        raise SystemExit(f"unknown task {task}")
    pd.DataFrame(rows).to_csv(results_tables(f"bb_pad_{task}.csv"), index=False)
    with open(log_path(f"15_{task}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


TASKS = ["E1_betablocker", "E1_CCB", "E1_diuretic", "E1b_athero", "E2_subtype", "E3_anyamp"]

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["combine"]:
        allr = pd.concat([pd.read_csv(results_tables(f"bb_pad_{t}.csv")) for t in TASKS], ignore_index=True)
        allr.to_csv(results_tables("bb_pad_exploration.csv"), index=False)
        print(allr.to_string())
    else:
        for a in args:
            run(a)
