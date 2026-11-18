"""
17_sensitivity_review3.py
-------------------------
NEW 2026-11-13. Sensitivity and descriptive analyses for the open items of the third (independent)
simulated review. Pre-specified BEFORE running in notes/2026-11-13_review3_tier2_prespec.md; the
primary analysis is unchanged.

Usage:
  python 17_sensitivity_review3.py check                 default build reproduces data/ccw_*_*.parquet
  python 17_sensitivity_review3.py <scenario> [...]      one CSV per scenario (results/tables/review3_<scenario>.csv)
  python 17_sensitivity_review3.py descriptives          D1-D6, D8
  python 17_sensitivity_review3.py combine               -> results/tables/sensitivity_review3.csv

Scenarios (48-h grace; same CCW/IPCW/IPTW/Cox/patient-bootstrap machinery):
  C1_admission_physiology     first K (24 h) + ED triage SBP/DBP (+ missing indicators) in both models
  C2a_exclude_early_limb_proc admissions with a limb procedure on admission day or the next day excluded
  C2b_early_proc_in_ipcw      early revascularization / amputation / debridement indicators in the IPCW model
  C3_admin_censoring_365d     follow-up truncated at day 365 after admission (all outcomes)
  C4_symmetric_comparator     ACEI/ARB without X vs X without ACEI/ARB
  C5_bootstrap_reestimated    weights re-estimated in every bootstrap replicate (primary specification)
  C6_ipcw_only                primary datasets weighted by the censoring weight alone
"""
import os
import sys
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from importlib import import_module
ccw_mod = import_module("05_ccw_clone_censor")
primary = import_module("06_primary_analysis")

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

COMPS = list(ccw_mod.COMPARATORS)
OUTS = list(ccw_mod.OUTCOMES)
G = GRACE_HOURS_PRIMARY
CLS_PREFIX = {"CCB": "ccb", "betablocker": "betablocker", "diuretic": "diuretic"}

sc = pd.read_parquet(data("sensitivity_covariates.parquet"))
r3 = pd.read_parquet(data("review3_covariates.parquet"))
el = pd.read_parquet(data("eligible_cohort.parquet"))
out = pd.read_parquet(data("outcomes.parquet"))
extra = sc.merge(r3.drop(columns=["subject_id"]), on="hadm_id", how="left").merge(
    el[["hadm_id", "hospital_expire_flag", "los_hours"]], on="hadm_id", how="left")
extra = extra.drop(columns=["ccb_first_h", "betablocker_first_h", "diuretic_first_h"])  # passed as drug flags (C4)
extra["potassium_unknown"] = extra["first_potassium_24h"].isna().astype(int)
extra["ed_bp_unknown"] = extra["ed_sbp"].isna().astype(int)
first_flags = r3[["hadm_id", "ccb_first_h", "betablocker_first_h", "diuretic_first_h"]]

# C3: follow-up truncated at day 365 after admission, all three outcomes
H = 365.0
c3 = pd.DataFrame({"hadm_id": out["hadm_id"]})
for tcol, ecol, _ in ccw_mod.OUTCOMES.values():
    c3[ecol] = out[ecol].astype(bool) & (out[tcol] <= H)
    c3[tcol] = out[tcol].clip(upper=H)

BASE_CONF = list(primary.CONFOUNDERS_BASE)
IPCW_BASE = list(ccw_mod.IPCW_COVARIATES)
PHYS = ["first_potassium_24h", "potassium_unknown", "ed_sbp", "ed_dbp", "ed_bp_unknown"]
EARLY = ["early_revasc", "early_amp", "early_debrid"]

SCENARIOS = {
    "C1_admission_physiology": dict(ipcw_covariates=IPCW_BASE + PHYS, iptw_conf=BASE_CONF + PHYS),
    "C2a_exclude_early_limb_proc": dict(admission_filter=lambda b: b["early_limb_proc"] == 0),
    "C2b_early_proc_in_ipcw": dict(ipcw_covariates=IPCW_BASE + EARLY),
    "C3_admin_censoring_365d": dict(outcome_override=c3),
    "C4_symmetric_comparator": dict(symmetric=True),
    "C5_bootstrap_reestimated": dict(special="reest"),
    "C6_ipcw_only": dict(special="ipcw_only"),
}


def build(comparison, outcome, spec, return_all=False):
    df = ccw_mod.build_ccw_dataset(G, comparison, outcome, log=log, extra_covariates=extra,
                                   ipcw_covariates=spec.get("ipcw_covariates"),
                                   admission_filter=spec.get("admission_filter"),
                                   outcome_override=spec.get("outcome_override"),
                                   extra_drug_flags=first_flags if spec.get("symmetric") else None,
                                   symmetric=spec.get("symmetric", False), return_all_clones=return_all)
    df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
    conf = [c for c in spec.get("iptw_conf", BASE_CONF) if c in df.columns]
    conf += [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
    if return_all:
        return df, conf
    attrs = dict(df.attrs)
    df = primary.add_combined_weight(df, conf)
    df.attrs.update(attrs)
    return df, conf


def n_imbalanced(df, conf, wcol="combined_weight"):
    t = df.treated == 1
    return int(sum(abs(primary.smd(df.loc[t, c].values, df.loc[~t, c].values, df.loc[t, wcol].values,
                                   df.loc[~t, wcol].values)) >= 0.10 for c in conf))


def weights_from_clones(allc, conf):
    """Re-fit the primary IPCW (per arm, at-risk clones) and IPTW models on a clone set; returns the
    retained clones with combined_weight. Mirrors 05.build_ccw_dataset + 06.add_combined_weight."""
    model_cols = allc.attrs["model_cols"]
    X = allc[model_cols].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    ipcw = np.ones(len(allc))
    shared = allc["shared_pre_grace"].values == 1
    for arm_val in (1, 0):
        at_risk = (allc["arm_is_aceiarb"].values == arm_val) & ~shared
        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
        pipe.fit(X[at_risk], allc["censored"].values[at_risk])
        ipcw[at_risk] = 1.0 / np.clip(1 - pipe.predict_proba(X[at_risk])[:, 1], 0.01, 0.99)
    lo, hi = np.percentile(ipcw[~shared], [1, 99])
    ipcw[~shared] = np.clip(ipcw[~shared], lo, hi)
    ret = allc.assign(ipcw_weight=ipcw)
    ret = ret[ret["censored"] == 0].copy()
    return primary.add_combined_weight(ret, conf)


def reest_bootstrap(comparison, outcome):
    allc, conf = build(comparison, outcome, {}, return_all=True)
    n_adm = allc.attrs["n_admissions"]
    point = weights_from_clones(allc, conf)
    cph = CoxPHFitter().fit(point[["treated", "event", "time", "subject_id", "combined_weight"]], duration_col="time",
                            event_col="event", weights_col="combined_weight", cluster_col="subject_id",
                            robust=True, formula="treated")
    hr = cph.hazard_ratios_["treated"]
    rt, rc = primary.arm_risks(point, "combined_weight")
    rng = np.random.default_rng(SEED)
    pids = allc["subject_id"].unique()
    hrs, rds, fails = [], [], 0
    for b in range(N_BOOT):
        cnt = pd.Series(rng.choice(pids, size=len(pids), replace=True)).value_counts()
        mult = allc["subject_id"].map(cnt).fillna(0).astype(int)
        rep = allc.loc[allc.index.repeat(mult)].reset_index(drop=True)
        rep.attrs["model_cols"] = allc.attrs["model_cols"]
        try:
            rw = weights_from_clones(rep, conf)
            cb = CoxPHFitter().fit(rw[["treated", "event", "time", "combined_weight"]], duration_col="time",
                                   event_col="event", weights_col="combined_weight", robust=True, formula="treated")
            hrs.append(cb.hazard_ratios_["treated"])
            a, c = primary.arm_risks(rw, "combined_weight")
            rds.append(a - c)
        except Exception:
            fails += 1
    lo, hi = np.percentile(hrs, [2.5, 97.5])
    rlo, rhi = np.percentile(rds, [2.5, 97.5])
    return point, conf, {"HR": hr, "ci_lo": lo, "ci_hi": hi, "cox_p": cph.summary.loc["treated", "p"],
                         "rd_1y": rt - rc, "rd_ci_lo": rlo, "rd_ci_hi": rhi, "n_boot": len(hrs), "n_fail": fails,
                         "n_adm": n_adm}


def run_scenario(name):
    spec = SCENARIOS[name]
    rows = []
    for comparison in COMPS:
        for outcome in OUTS:
            log(f"\n=== {name} / {comparison} / {outcome} ===")
            if spec.get("special") == "reest":
                df, conf, fit = reest_bootstrap(comparison, outcome)
                wcol = "combined_weight"
                n_adm = fit["n_adm"]
            else:
                df, conf = build(comparison, outcome, spec)
                wcol = "ipcw_weight" if spec.get("special") == "ipcw_only" else "combined_weight"
                fit = primary.cox_hr_bootstrap(df, weight_col=wcol)
                n_adm = df.attrs["n_admissions"]
            t = df.treated == 1
            n_imb = n_imbalanced(df, conf, wcol)
            row = {"scenario": name, "comparison": comparison, "outcome": outcome, "n_admissions": n_adm,
                   "n_treated": int(t.sum()), "n_control": int((~t).sum()),
                   "events_treated": int(df.loc[t, "event"].sum()), "events_control": int(df.loc[~t, "event"].sum()),
                   "HR": fit["HR"], "HR_CI_lower": fit["ci_lo"], "HR_CI_upper": fit["ci_hi"], "cox_p_value": fit["cox_p"],
                   "rd_1y": fit["rd_1y"], "rd_1y_CI_lower": fit["rd_ci_lo"], "rd_1y_CI_upper": fit["rd_ci_hi"],
                   "n_covariates_imbalanced": n_imb, "n_covariates": len(conf),
                   "max_weight": float(df[wcol].max()), "n_boot_successful": fit["n_boot"]}
            log(f"  -> HR {fit['HR']:.3f} [{fit['ci_lo']:.3f}, {fit['ci_hi']:.3f}] p={fit['cox_p']:.3f}; "
                f"RD {100*fit['rd_1y']:.1f} [{100*fit['rd_ci_lo']:.1f}, {100*fit['rd_ci_hi']:.1f}]; "
                f"n {row['n_treated']}/{row['n_control']}, events {row['events_treated']}/{row['events_control']}, "
                f"imbalanced {n_imb}/{len(conf)}, boot {fit['n_boot']}")
            rows.append(row)
    pd.DataFrame(rows).to_csv(results_tables(f"review3_{name}.csv"), index=False)
    with open(log_path(f"17_{name}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


def check():
    """Default arguments (plus the new optional ones at their defaults) reproduce the saved primary datasets."""
    for comparison in COMPS:
        for outcome in OUTS:
            new = ccw_mod.build_ccw_dataset(G, comparison, outcome, log=lambda m: None)
            old = pd.read_parquet(data(f"ccw_{comparison}_{outcome}.parquet"))
            same = (len(new) == len(old) and (new["hadm_id"].values == old["hadm_id"].values).all()
                    and np.allclose(new["ipcw_weight"].values, old["ipcw_weight"].values)
                    and np.allclose(new["time"].values, old["time"].values)
                    and (new["event"].values == old["event"].values).all())
            log(f"check {comparison}/{outcome}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(new)} rows)")
            assert same


def kish(w):
    w = np.asarray(w, float)
    return w.sum() ** 2 / (w ** 2).sum()


def descriptives():
    ex = extra.set_index("hadm_id")
    drugs = pd.read_parquet(data("drug_exposure.parquet")).set_index("hadm_id")
    d1, d2ess, d2p, d3, d4, d8 = [], [], [], [], [], []
    for comparison in COMPS:
        cp = CLS_PREFIX[comparison]
        for outcome in OUTS:
            df, _ = primary.load_weighted(comparison, outcome)
            for arm, lab in [(1, "ACEI/ARB"), (0, comparison)]:
                w = df.loc[df.treated == arm, "combined_weight"]
                d2ess.append({"comparison": comparison, "outcome": outcome, "arm": lab, "n_retained": len(w),
                              "kish_ess": kish(w)})
            if outcome == "mortality":
                for c in ["early_revasc", "early_amp", "early_debrid", "early_limb_proc", "index_revasc",
                          "ed_linked", "medrecon_available", "pre_aceiarb", "pre_ccb", "pre_betablocker",
                          "pre_diuretic", "ed_sbp", "ed_dbp", "last24_aceiarb", f"last24_{cp}", "aceiarb_after48h",
                          "hospital_expire_flag", "los_hours"]:
                    df[c] = df["hadm_id"].map(ex[c])
                fol = df[df["shared_pre_grace"] == 0]
                for arm, lab in [(1, "ACEI/ARB"), (0, comparison)]:
                    x = df[df.treated == arm]
                    f = fol[fol.treated == arm]
                    p = f[(f.hospital_expire_flag == 0) & (f.los_hours > G)]
                    wv = p["combined_weight"]
                    anchor = "last24_aceiarb" if arm == 1 else f"last24_{cp}"
                    row = {"comparison": comparison, "arm": lab, "n_following_alive_los_gt48h": len(p),
                           "pct_anchor_last24h": 100 * p[anchor].mean(),
                           "pct_anchor_last24h_weighted": 100 * np.average(p[anchor], weights=wv)}
                    if arm == 0:
                        row.update({"pct_aceiarb_after48h": 100 * p["aceiarb_after48h"].mean(),
                                    "pct_aceiarb_after48h_weighted": 100 * np.average(p["aceiarb_after48h"], weights=wv),
                                    "pct_aceiarb_last24h": 100 * p["last24_aceiarb"].mean(),
                                    "pct_aceiarb_last24h_weighted": 100 * np.average(p["last24_aceiarb"], weights=wv)})
                    d1.append(row)
                    mr = f[f.medrecon_available == 1]
                    d3.append({"comparison": comparison, "arm": lab, "n_following": len(f),
                               "pct_ed_linked": 100 * f.ed_linked.mean(), "pct_medrecon": 100 * f.medrecon_available.mean(),
                               "n_medrecon": len(mr),
                               **{f"pct_pre_{c}": 100 * mr[f"pre_{c}"].mean() for c in ["aceiarb", "ccb", "betablocker", "diuretic"]},
                               "pct_ed_sbp_available": 100 * f.ed_sbp.notna().mean(),
                               "ed_sbp_mean": f.ed_sbp.mean(), "ed_dbp_mean": f.ed_dbp.mean()})
                    d4.append({"comparison": comparison, "arm": lab, "n_retained": len(x),
                               **{f"pct_{c}": 100 * x[c].mean() for c in ["early_revasc", "early_amp", "early_debrid",
                                                                           "early_limb_proc", "index_revasc"]},
                               **{f"pct_{c}_weighted": 100 * np.average(x[c], weights=x.combined_weight)
                                  for c in ["early_revasc", "early_amp", "early_debrid", "early_limb_proc", "index_revasc"]}})
            if outcome == "afs":
                allc, _ = build(comparison, "afs", {}, return_all=True)
                allc["contra"] = allc["hadm_id"].map(ex["contraindication_ras"])
                ar = allc[allc["shared_pre_grace"] == 0]
                for arm, lab in [(1, "ACEI/ARB"), (0, comparison)]:
                    for ci in (0, 1):
                        pv = ar.loc[(ar.arm_is_aceiarb == arm) & (ar.contra == ci), "p_uncensored"]
                        d2p.append({"comparison": comparison, "arm": lab, "baseline_contraindication": ci, "n": len(pv),
                                    "p_min": pv.min(), "p_1pct": np.percentile(pv, 1), "p_median": pv.median()})
            if outcome == "amputation":
                o = out.set_index("hadm_id")
                df["idx_amp"] = (df["hadm_id"].map(o["amp_hadm_id"]) == df["hadm_id"])
                df["later_adm"] = df["hadm_id"].map(ex["later_adm_365d"])
                df["alive_disch"] = df["hadm_id"].map(ex["hospital_expire_flag"]) == 0
                ev1y = (df["event"] == 1) & (df["time"] <= 365)
                for arm, lab in [(1, "ACEI/ARB"), (0, comparison)]:
                    m = df.treated == arm
                    w = df.loc[m, "combined_weight"]
                    idx = np.average((ev1y & df.idx_amp)[m], weights=w)
                    post = np.average((ev1y & ~df.idx_amp)[m], weights=w)
                    a = m & df.alive_disch
                    pl = np.average(df.loc[a, "later_adm"], weights=df.loc[a, "combined_weight"])
                    d8.append({"comparison": comparison, "arm": lab, "risk_index_stay": idx, "risk_post_discharge": post,
                               "risk_observed": idx + post, "p_later_admission": pl, "risk_corrected": idx + post / pl})
    # D2: Table 2 columns for the full cohort and admissions with no studied class within 48 h
    cov = pd.read_parquet(data("covariates.parquet"))
    cov["gender_male"] = (cov["gender"] == "M").astype(int)
    for r in ["White", "Black", "Hispanic", "Asian", "Other/Unknown"]:
        cov[f"race_{r}"] = (cov["race_simplified"] == r).astype(int)
    nocls = ~drugs.loc[cov.hadm_id, [f"{c}_{G}h" for c in ["aceiarb", "ccb", "betablocker", "diuretic"]]].any(axis=1).values
    t2 = {}
    for lab, sub in [("Full cohort", cov), ("No studied class within 48 h", cov[nocls])]:
        col = {"Admissions, n": f"{len(sub)}", "Patients, n": f"{sub.subject_id.nunique()}"}
        for v in ["age_at_admission", "gender_male", "race_White", "race_Black", "race_Hispanic", "race_Asian",
                  "race_Other/Unknown", "emergency_admission", "icu_stay_flag", "mi", "chf", "cerebrovascular",
                  "chronic_pulmonary", "renal", "sepsis", "charlson_comorbidity_index", "baseline_creatinine",
                  "creatinine_unknown", "baseline_sbp", "baseline_dbp", "bp_unknown"]:
            x = sub[v].astype(float)
            cont = v in {"age_at_admission", "charlson_comorbidity_index", "baseline_creatinine", "baseline_sbp", "baseline_dbp"}
            col[v] = f"{x.mean():.1f} ({x.std():.1f})" if cont else f"{100 * x.mean():.1f}"
        t2[lab] = col
    # D5 patient-level counts; D6 add-on classes in the ACEI/ARB strategy
    ae = out[out.afs_eligible]
    d5 = pd.DataFrame([
        {"outcome": "death", "admissions": int(out.mortality_event.sum()),
         "patients": out.loc[out.mortality_event.astype(bool), "subject_id"].nunique()},
        {"outcome": "major amputation", "admissions": int(ae.amputation_only_event.sum()),
         "patients": ae.loc[ae.amputation_only_event.astype(bool), "subject_id"].nunique()},
        {"outcome": "AFS event", "admissions": int(ae.afs_event.sum()),
         "patients": ae.loc[ae.afs_event.astype(bool), "subject_id"].nunique()},
        {"outcome": "all (denominator)", "admissions": len(out), "patients": out.subject_id.nunique()}])
    a = drugs[drugs[f"aceiarb_{G}h"]]
    d6 = pd.DataFrame([{"n_aceiarb_48h": len(a), **{f"pct_{c}": 100 * a[f"{c}_{G}h"].mean()
                                                    for c in ["ccb", "betablocker", "diuretic"]},
                        "pct_monotherapy": 100 * (~a[[f"{c}_{G}h" for c in ["ccb", "betablocker", "diuretic"]]].any(axis=1)).mean()}])
    for nm, obj in [("D1_persistence", pd.DataFrame(d1)), ("D2_ess", pd.DataFrame(d2ess)),
                    ("D2_p_uncensored_by_contraindication", pd.DataFrame(d2p)),
                    ("D2_table2_extra_columns", pd.DataFrame(t2).rename_axis("variable").reset_index()),
                    ("D3_ed_by_arm", pd.DataFrame(d3)), ("D4_early_procedures_by_arm", pd.DataFrame(d4)),
                    ("D5_patient_level_counts", d5), ("D6_addon_classes", d6), ("D8_qba_amputation", pd.DataFrame(d8))]:
        obj.to_csv(results_tables(f"review3_{nm}.csv"), index=False)
        log(f"\n{nm}\n" + obj.to_string())
    with open(log_path("17_descriptives.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["check"]:
        check()
    elif args == ["descriptives"]:
        descriptives()
    elif args == ["combine"]:
        allr = pd.concat([pd.read_csv(results_tables(f"review3_{s}.csv")) for s in SCENARIOS], ignore_index=True)
        allr.to_csv(results_tables("sensitivity_review3.csv"), index=False)
        print(allr.to_string())
    else:
        for a in args:
            run_scenario(a)
