"""
05_ccw_clone_censor.py
------------------------
NEW 2026-08-31 (REVISION 3). Clone-Censor-Weight (CCW) implementation,
adapted from project_001's precedent (project_001_atas-antihypertensive-
tte/scripts/05_ccw.py) to this project's anchor-drug, WINDOW-based (not
race/first-event-based) treatment definition (CLAUDE.md REVISION 2).

Mechanics (see notes/2026-08-31_ccw_redesign_decision.md for the full
diagnosis/derivation):
  1. CLONE every eligible admission into two rows per comparison: an
     "ACEI_ARB" clone and a comparator-X clone. Both start follow-up at
     T0 = admittime.
  2. Both clones share the SAME (event, time) outcome pair for as long as
     the outcome time falls WITHIN the grace period (<= g) -- this is the
     actual bias fix: a death/AFS-event before the grace window closes
     can't yet be attributed to one strategy over the other (REVISION 2's
     arm definition is only resolved AT the end of the grace window), so
     it counts as an event for BOTH clones' risk sets, recovering exactly
     the admissions the old 48h-eligibility-landmark design silently
     dropped.
  3. For admissions whose outcome time is AFTER the grace period, resolve
     arm membership at the grace boundary (REVISION 2's existing
     definition): the clone matching the resolved arm continues
     uncensored to the real outcome time; the other clone is artificially
     censored AT the grace boundary (deviated from its assigned
     strategy).
  4. WEIGHT (IPCW): single cross-sectional logistic model for
     P(censored=1 | baseline covariates), fit SEPARATELY PER CLONE ARM and
     ONLY on clones still at risk of deviation-censoring at the grace
     boundary (outcome time > g). Retained at-risk clones get weight
     1/P(uncensored), trimmed at 1st/99th percentile (of the at-risk
     weights); clones whose outcome occurs within the grace period
     (shared events) are structurally uncensorable and get weight 1.
     Unstabilized, so each arm's pseudo-population represents the full
     eligible cohort. Same non-time-varying rigor level as project_001's
     precedent (one censoring time point, not full time-varying
     g-methods). Censored clones are dropped from the final analysis
     dataset (equivalent to zero-weighting them).
     CORRECTED 2026-10-01: the earlier version fit one pooled model on ALL
     clones (incl. uncensorable shared-event clones), with an arm term and
     an arm-share numerator. That gave shared early events large,
     arm-asymmetric weights (e.g. CCB/AFS median 25.6 comparator vs 10.7
     ACEI/ARB; early events ~66% vs ~40% of weighted mass vs 1.8% of
     admissions). See notes/2026-10-01_ipcw_shared_event_weight_issue.md.

Usable as a module (build_ccw_dataset) by 06 (primary, grace=48h) and 07
(sensitivity, grace=24h/72h) so the logic is written once.

Output when run standalone: one CCW clone dataset per (comparison,
outcome) at the PRIMARY grace period, in data/ccw_{comparison}_{outcome}.parquet
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from config import *

COMPARATORS = {"CCB": "ccb", "betablocker": "betablocker", "diuretic": "diuretic"}
# Comparator-definition sensitivity (script 10, added 2026-11-10): not iterated by the
# primary loops in 05/06/07/08/09, only built on request.
SENSITIVITY_COMPARATORS = {"CCB_DHP": "ccbdhp", "diuretic_thiazide": "thiazidenoloop"}
# Exploratory (2026-11-10, notes/2026-11-10_betablocker_pad_exploration_prespec.md); the
# flags come from data/pad_bb_anyamp.parquet via build_ccw_dataset(extra_drug_flags=...).
EXPLORATORY_COMPARATORS = {"BB_vasodilating": "bbvasoonly", "BB_nonvasodilating": "bbnonvasoonly"}
ALL_COMPARATORS = {**COMPARATORS, **SENSITIVITY_COMPARATORS, **EXPLORATORY_COMPARATORS}
# CORRECTION 2026-11-10 (notes/2026-11-10_grace_deviation_fix.md): drugs whose order makes a
# COMPARATOR clone deviate from its strategy (first order times from script 02, *_first_h).
# The ACEI/ARB strategy cannot be violated inside the grace period.
DEVIATION_DRUGS = {"CCB": ["aceiarb"], "betablocker": ["aceiarb"], "diuretic": ["aceiarb"],
                   "CCB_DHP": ["aceiarb"], "diuretic_thiazide": ["aceiarb", "loop"],
                   "BB_vasodilating": ["aceiarb", "bbnonvaso"], "BB_nonvasodilating": ["aceiarb", "bbvaso"]}
OUTCOMES = {
    "afs": ("afs_days_from_t0", "afs_event", "afs_eligible"),
    "mortality": ("mortality_days_from_t0", "mortality_event", None),
    "amputation": ("amputation_only_days_from_t0", "amputation_only_event", "afs_eligible"),
}

IPCW_COVARIATES = ["age_at_admission", "gender_male", "mi", "chf", "cerebrovascular",
                    "chronic_pulmonary", "renal", "sepsis", "charlson_comorbidity_index",
                    "baseline_creatinine", "creatinine_unknown", "baseline_sbp",
                    "baseline_dbp", "bp_unknown", "emergency_admission", "icu_stay_flag"]


def build_ccw_dataset(grace_hours, comparison, outcome_name, log=print, ipcw_covariates=None, include_race=True,
                      extra_covariates=None, admission_filter=None, outcome_override=None,
                      extra_drug_flags=None, symmetric=False, return_all_clones=False):
    """Returns the uncensored CCW clone dataset (one row per surviving
    clone) for the given grace period / comparator / outcome, with an
    `ipcw_weight` column, ready for merging with covariates and a
    baseline-confounding IPTW weight in the primary-analysis script.

    Optional (added 2026-11-10 for script 12; defaults reproduce the primary analysis):
      extra_covariates  DataFrame keyed on hadm_id, merged onto the admissions
      admission_filter  callable(base) -> boolean mask, applied before cloning
      outcome_override  DataFrame with hadm_id + this outcome's time/event columns,
                        replacing the values from outcomes.parquet
      extra_drug_flags  DataFrame keyed on hadm_id with extra <prefix>_<grace>h columns
                        (script 15, exploratory beta-blocker subtype comparators)
    Added 2026-11-13 for script 17 (review v3; defaults unchanged):
      symmetric          ACEI/ARB strategy = ACEI/ARB WITHOUT the comparator class; an ACEI/ARB clone
                         also deviates when the comparator class is ordered before its in-grace event
                         (needs <comp>_first_h, passed through extra_drug_flags)
      return_all_clones  return every clone (incl. censored) with p_uncensored, for weight
                         re-estimation inside the bootstrap"""
    eligible = pd.read_parquet(data("eligible_cohort.parquet"))
    drugs = pd.read_parquet(data("drug_exposure.parquet"))
    if extra_drug_flags is not None:
        drugs = drugs.merge(extra_drug_flags.drop(columns=["subject_id"], errors="ignore"), on="hadm_id", how="left")
    outcomes = pd.read_parquet(data("outcomes.parquet"))
    covariates = pd.read_parquet(data("covariates.parquet"))

    comp_prefix = ALL_COMPARATORS[comparison]
    time_col, event_col, eligible_flag = OUTCOMES[outcome_name]

    base = eligible[["hadm_id", "subject_id"]].merge(
        drugs[["hadm_id", f"aceiarb_{grace_hours}h", f"{comp_prefix}_{grace_hours}h"]
              + [f"{d}_first_h" for d in DEVIATION_DRUGS[comparison]]
              + ([f"{comp_prefix}_first_h"] if symmetric else [])],
        on="hadm_id", how="left",
    )
    base = base.merge(outcomes[["hadm_id", time_col, event_col] + ([eligible_flag] if eligible_flag else [])],
                       on="hadm_id", how="left")
    base = base.merge(covariates, on=["hadm_id", "subject_id"], how="left")
    if outcome_override is not None:
        ov = outcome_override.set_index("hadm_id")
        base[time_col] = base["hadm_id"].map(ov[time_col])
        base[event_col] = base["hadm_id"].map(ov[event_col])
    if extra_covariates is not None:
        base = base.merge(extra_covariates.drop(columns=["subject_id"], errors="ignore"), on="hadm_id", how="left")

    n_before = len(base)
    if eligible_flag:
        base = base[base[eligible_flag]].copy()
    if admission_filter is not None:
        base = base[admission_filter(base)].copy()
        log(f"  admission_filter applied: {len(base):,} admissions kept")
    log(f"[{comparison}/{outcome_name}/g={grace_hours}h] Admissions entering CCW "
        f"({'AFS-eligible subset' if eligible_flag else 'full cohort'}): {len(base):,} / {n_before:,}")

    base["gender_male"] = (base["gender"] == "M").astype(int)
    race_dummies = pd.get_dummies(base["race_simplified"], prefix="race", drop_first=True).astype(int)
    base = pd.concat([base, race_dummies], axis=1)

    aceiarb_col = f"aceiarb_{grace_hours}h"
    comp_col = f"{comp_prefix}_{grace_hours}h"
    g_days = grace_hours / 24.0

    if symmetric:
        confirmed_arm = np.where(base[aceiarb_col] & ~base[comp_col], "ACEI_ARB",
                         np.where(base[comp_col] & ~base[aceiarb_col], comparison, "neither"))
    else:
        confirmed_arm = np.where(base[aceiarb_col], "ACEI_ARB",
                         np.where(base[comp_col] & ~base[aceiarb_col], comparison, "neither"))
    shared_mask = base[time_col].values <= g_days
    # CORRECTION 2026-11-10: a comparator clone whose first deviating order (within the grace
    # period) comes strictly BEFORE its in-grace outcome had already left its strategy, so it is
    # a deviator (censored), not a shared event. The ACEI/ARB clone keeps the event.
    first_dev_h = base[[f"{d}_first_h" for d in DEVIATION_DRUGS[comparison]]].min(axis=1).values
    dev_days = np.where(first_dev_h <= grace_hours, first_dev_h / 24.0, np.inf)
    dev_before_event = shared_mask & (dev_days < base[time_col].values)
    arm_shared = {"ACEI_ARB": shared_mask, comparison: shared_mask & ~dev_before_event}
    if symmetric:
        acei_dev_h = base[f"{comp_prefix}_first_h"].values
        acei_dev_days = np.where(acei_dev_h <= grace_hours, acei_dev_h / 24.0, np.inf)
        arm_shared["ACEI_ARB"] = shared_mask & ~(acei_dev_days < base[time_col].values)

    clone_rows = []
    for arm_name in ["ACEI_ARB", comparison]:
        clone = base.copy()
        clone["clone_arm"] = arm_name
        is_confirmed = confirmed_arm == arm_name
        sh = arm_shared[arm_name]
        clone["time"] = np.where(sh, base[time_col],
                          np.where(is_confirmed, base[time_col], g_days))
        clone["event"] = np.where(sh, base[event_col],
                           np.where(is_confirmed, base[event_col], 0)).astype(int)
        # Artificial (deviation) censoring: not a (non-deviated) in-grace event AND this
        # clone doesn't match the resolved arm.
        clone["censored"] = ((~sh) & (~is_confirmed)).astype(int)
        clone_rows.append(clone)
    clones = pd.concat(clone_rows, ignore_index=True)
    n_dev_before_event = int((dev_before_event & base[event_col].astype(bool).values).sum())
    log(f"  In-grace events after a deviating order (comparator clone censored, event kept in "
        f"the ACEI/ARB clone): {n_dev_before_event:,}")

    n_shared_events = (shared_mask & base[event_col].astype(bool)).sum()
    log(f"  Shared pre-grace events (counted in BOTH clones' risk sets, the "
        f"REVISION-3 bias fix): {n_shared_events:,}")
    log(f"  Clones: {len(clones):,} ({len(clones)//2:,} admissions x 2). "
        f"Artificially censored (deviators): {clones.censored.sum():,} "
        f"({100*clones.censored.mean():.1f}%)")

    # --- IPCW (corrected 2026-10-01): per-arm model of P(censored=1 |
    # baseline covariates), fit only on clones at risk of deviation-
    # censoring at the grace boundary; shared-event clones get weight 1 ---
    clones["arm_is_aceiarb"] = (clones["clone_arm"] == "ACEI_ARB").astype(int)
    clones["shared_pre_grace"] = np.concatenate([arm_shared["ACEI_ARB"], arm_shared[comparison]]).astype(int)
    # ipcw_covariates/include_race: covariate subsets for the 09 confounding
    # decomposition; defaults reproduce the primary analysis.
    cov_list = IPCW_COVARIATES if ipcw_covariates is None else ipcw_covariates
    model_cols = [c for c in cov_list if c in clones.columns] + (list(race_dummies.columns) if include_race else [])
    X = clones[model_cols].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    ipcw = np.ones(len(clones))
    p_unc_all = np.full(len(clones), np.nan)
    p_unc_summary = {}
    for arm_val in [1, 0]:
        at_risk = ((clones["arm_is_aceiarb"] == arm_val) & (clones["shared_pre_grace"] == 0)).values
        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
        pipe.fit(X[at_risk], clones.loc[at_risk, "censored"].values)
        prob_uncensored = np.clip(1 - pipe.predict_proba(X[at_risk])[:, 1], 0.01, 0.99)
        ipcw[at_risk] = 1.0 / prob_uncensored
        p_unc_all[at_risk] = prob_uncensored
        p_unc_summary["ACEI_ARB" if arm_val == 1 else comparison] = (
            float(prob_uncensored.min()), float(np.percentile(prob_uncensored, 1)), float(np.median(prob_uncensored)))
    at_risk_all = clones["shared_pre_grace"].values == 0
    lo, hi = np.percentile(ipcw[at_risk_all], [1, 99])
    ipcw[at_risk_all] = np.clip(ipcw[at_risk_all], lo, hi)
    clones["ipcw_weight"] = ipcw
    if return_all_clones:
        clones["p_uncensored"] = p_unc_all
        clones.attrs["model_cols"] = model_cols
        clones.attrs["n_admissions"] = int(len(base))
        return clones

    ccw = clones[clones["censored"] == 0].copy()
    # Diagnostics for script 12 (in-memory only; not written to parquet)
    ccw.attrs["p_uncensored"] = p_unc_summary
    ccw.attrs["n_admissions"] = int(len(base))
    ccw.attrs["n_shared"] = int(shared_mask.sum())
    ccw.attrs["n_shared_events"] = int(n_shared_events)
    ccw.attrs["n_dev_before_event"] = n_dev_before_event
    ccw.attrs["n_follow"] = {k: int(((confirmed_arm == k) & ~shared_mask).sum())
                             for k in ["ACEI_ARB", comparison, "neither"]}
    log(f"  Uncensored (retained) clones: {len(ccw):,} "
        f"(ACEI_ARB={ (ccw.clone_arm=='ACEI_ARB').sum():,}, {comparison}={(ccw.clone_arm==comparison).sum():,})")
    for arm_name in ["ACEI_ARB", comparison]:
        w = ccw.loc[ccw.clone_arm == arm_name, "ipcw_weight"]
        log(f"  IPCW {arm_name}: sum={w.sum():,.0f} (vs {len(base):,} admissions), "
            f"median={w.median():.2f}, p99={w.quantile(0.99):.2f}, max={w.max():.2f}")
    return ccw


if __name__ == "__main__":
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 70)
    log(f"CCW clone-censor-weight, PRIMARY grace period = {GRACE_HOURS_PRIMARY}h")
    log("=" * 70)

    for comparison in COMPARATORS:
        for outcome_name in OUTCOMES:
            ccw = build_ccw_dataset(GRACE_HOURS_PRIMARY, comparison, outcome_name, log=log)
            ccw.to_parquet(data(f"ccw_{comparison}_{outcome_name}.parquet"), index=False)
            log(f"  Saved: data/ccw_{comparison}_{outcome_name}.parquet\n")

    with open(log_path("05_ccw_clone_censor.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
