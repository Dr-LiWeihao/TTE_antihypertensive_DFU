"""
12_sensitivity_review2.py
--------------------------
NEW 2026-11-10. Sensitivity and descriptive analyses requested after the second simulated
pre-submission review. Pre-specified BEFORE running in
notes/2026-11-10_review2_sensitivity_prespec.md; the primary analysis is unchanged.

Usage:
  python 12_sensitivity_review2.py <scenario> [<scenario> ...]   one CSV per scenario
  python 12_sensitivity_review2.py descriptives                  A1a, A2a, B9 (no bootstrap)
  python 12_sensitivity_review2.py combine                       -> sensitivity_review2.csv

Scenarios (all: 48-h grace, same CCW/IPCW/IPTW/Cox/patient-bootstrap machinery):
  A1b_no_contraindication     exclude first K > 5.5 mmol/L or creatinine > 3.0 mg/dL (<=24 h)
  A1c_any_class_48h           admissions with >=1 studied class ordered within 48 h
                              (conditions on early treatment -- a bound, not an estimand)
  A2b_index_stay_only         AFS / amputation with follow-up ending at index discharge
  B4_first_admission          first eligible admission per patient
  B5_baseline_only_covariates sepsis dropped; ICU stay -> ICU admission within 48 h
  B6_calendar_era             approximate admission era added to both models
  B7_extended_censoring_model 0-48 h creatinine change and max potassium in the IPCW model
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
    print(msg, flush=True)
    log_lines.append(str(msg))

COMPS = list(ccw_mod.COMPARATORS)
G = GRACE_HOURS_PRIMARY

sc = pd.read_parquet(data("sensitivity_covariates.parquet"))
drugs = pd.read_parquet(data("drug_exposure.parquet"))
anycls = drugs[["hadm_id"]].copy()
anycls["any_class_48h"] = drugs[[f"{c}_{G}h" for c in ["aceiarb", "ccb", "betablocker", "diuretic"]]].any(axis=1)
elig = pd.read_parquet(data("eligible_cohort.parquet"))
extra = sc.merge(anycls, on="hadm_id", how="left").merge(
    elig[["hadm_id", "hospital_expire_flag"]], on="hadm_id", how="left")
first_ids = set(elig.sort_values("admittime").groupby("subject_id")["hadm_id"].first())

# A2b: outcome override, follow-up ends at index discharge
out = pd.read_parquet(data("outcomes.parquet"))
los = (out["dischtime"] - out["admittime"]).dt.total_seconds() / 86400
inhosp_death = out["hospital_expire_flag"] == 1
idx_amp = out["amputation_only_event"] & (out["amp_hadm_id"] == out["hadm_id"])
amp_t = out["days_to_major_amp"].clip(upper=los)
death_t = out["mortality_days_from_t0"].clip(upper=los)
idx_override = pd.DataFrame({"hadm_id": out["hadm_id"]})
idx_override["amputation_only_event"] = idx_amp
idx_override["amputation_only_days_from_t0"] = np.where(idx_amp, amp_t, los)
afs_ev = inhosp_death | idx_amp
afs_t = np.where(idx_amp & inhosp_death, np.minimum(amp_t, death_t),
                 np.where(idx_amp, amp_t, np.where(inhosp_death, death_t, los)))
idx_override["afs_event"] = afs_ev
idx_override["afs_days_from_t0"] = afs_t

BASE_CONF = list(primary.CONFOUNDERS_BASE)
IPCW_BASE = list(ccw_mod.IPCW_COVARIATES)
ERA = [c for c in sc.columns if c.startswith("era_")]
B5 = [c for c in BASE_CONF if c not in ("sepsis", "icu_stay_flag")] + ["icu_within_48h"]
B7 = IPCW_BASE + ["crea_change_48h", "crea_change_48h_unknown", "max_potassium_48h", "max_potassium_48h_unknown"]

SCENARIOS = {
    "A1b_no_contraindication": dict(admission_filter=lambda b: b["contraindication_ras"] == 0),
    "A1c_any_class_48h": dict(admission_filter=lambda b: b["any_class_48h"].fillna(False).astype(bool)),
    "A2b_index_stay_only": dict(outcome_override=idx_override, outcomes=["afs", "amputation"]),
    "B4_first_admission": dict(admission_filter=lambda b: b["hadm_id"].isin(first_ids)),
    "B5_baseline_only_covariates": dict(ipcw_covariates=B5, iptw_conf=B5),
    "B6_calendar_era": dict(ipcw_covariates=IPCW_BASE + ERA, iptw_conf=BASE_CONF + ERA),
    "B7_extended_censoring_model": dict(ipcw_covariates=B7),
}


def build(comparison, outcome, spec):
    df = ccw_mod.build_ccw_dataset(G, comparison, outcome, log=log, extra_covariates=extra,
                                   ipcw_covariates=spec.get("ipcw_covariates"),
                                   admission_filter=spec.get("admission_filter"),
                                   outcome_override=spec.get("outcome_override"))
    df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
    conf = [c for c in spec.get("iptw_conf", BASE_CONF) if c in df.columns]
    conf += [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
    attrs = dict(df.attrs)
    df = primary.add_combined_weight(df, conf)
    df.attrs.update(attrs)
    return df, conf


def run_scenario(name):
    spec = SCENARIOS[name]
    rows = []
    for comparison in COMPS:
        for outcome in spec.get("outcomes", list(ccw_mod.OUTCOMES)):
            log(f"\n=== {name} / {comparison} / {outcome} ===")
            df, conf = build(comparison, outcome, spec)
            t = df.treated == 1
            n_imb = sum(abs(primary.smd(df.loc[t, c].values, df.loc[~t, c].values,
                                        df.loc[t, "combined_weight"].values,
                                        df.loc[~t, "combined_weight"].values)) >= 0.10 for c in conf)
            fit = primary.cox_hr_bootstrap(df)
            row = {"scenario": name, "comparison": comparison, "outcome": outcome,
                   "n_admissions": df.attrs["n_admissions"],
                   "n_treated": int(t.sum()), "n_control": int((~t).sum()),
                   "events_treated": int(df.loc[t, "event"].sum()), "events_control": int(df.loc[~t, "event"].sum()),
                   "HR": fit["HR"], "HR_CI_lower": fit["ci_lo"], "HR_CI_upper": fit["ci_hi"], "cox_p_value": fit["cox_p"],
                   "rd_1y": fit["rd_1y"], "rd_1y_CI_lower": fit["rd_ci_lo"], "rd_1y_CI_upper": fit["rd_ci_hi"],
                   "n_covariates_imbalanced": int(n_imb), "n_covariates": len(conf),
                   "max_combined_weight": float(df["combined_weight"].max()), "n_boot_successful": fit["n_boot"]}
            log(f"  -> HR {fit['HR']:.3f} [{fit['ci_lo']:.3f}, {fit['ci_hi']:.3f}] p={fit['cox_p']:.3f}; "
                f"n {row['n_treated']}/{row['n_control']}, events {row['events_treated']}/{row['events_control']}, "
                f"imbalanced {n_imb}/{len(conf)}")
            rows.append(row)
    pd.DataFrame(rows).to_csv(results_tables(f"sens_review2_{name}.csv"), index=False)
    with open(log_path(f"12_{name}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


def descriptives():
    """A1a (strategy-following, P(uncensored)), A2a (later same-hospital admission by arm),
    B9 (reasons for end of follow-up by arm). Primary specification, no bootstrap."""
    a1, a2, b9 = [], [], []
    for comparison in COMPS:
        for outcome in ccw_mod.OUTCOMES:
            df, _ = build(comparison, outcome, {})
            nf, pu = df.attrs["n_follow"], df.attrs["p_uncensored"]
            n = df.attrs["n_admissions"]
            a1.append({"comparison": comparison, "outcome": outcome, "n_admissions": n,
                       "n_event_within_grace": df.attrs["n_shared"],
                       "n_follow_aceiarb": nf["ACEI_ARB"], "n_follow_comparator": nf[comparison],
                       "n_follow_neither": nf["neither"], "pct_neither": 100 * nf["neither"] / n,
                       **{f"p_uncens_{k}_{s}": v for k, vals in pu.items()
                          for s, v in zip(["min", "p1", "median"], vals)},
                       "max_combined_weight_untrimmed": float(df["combined_weight_raw"].max())})
            for arm, lab in [(1, "ACEI/ARB"), (0, comparison)]:
                d = df[df.treated == arm]
                n_retained = len(d)
                ev = int(d["event"].sum())
                if outcome == "amputation":
                    o = out.set_index("hadm_id")
                    md = d["hadm_id"].map(o["mortality_days_from_t0"])
                    me = d["hadm_id"].map(o["mortality_event"]).astype(bool)
                    death = int(((d["event"] == 0) & me & (np.abs(d["time"] - md) < 1e-6)).sum())
                else:
                    death = 0
                b9.append({"comparison": comparison, "outcome": outcome, "arm": lab,
                           "clones": n, "artificially_censored_48h": n - n_retained,
                           "outcome_event": ev, "death_censoring": death,
                           "administrative_end": n_retained - ev - death})
            if outcome == "afs":
                d = df[df["hospital_expire_flag"] == 0]
                for arm, lab in [(1, "ACEI/ARB"), (0, comparison)]:
                    x = d[d.treated == arm]
                    a2.append({"comparison": comparison, "arm": lab, "n_alive_at_discharge": len(x),
                               "pct_later_admission_365d_unweighted": 100 * x["later_adm_365d"].mean(),
                               "pct_later_admission_365d_weighted":
                                   100 * np.average(x["later_adm_365d"], weights=x["combined_weight"])})
    for nm, rows in [("sens_review2_A1a_strategy_following.csv", a1),
                     ("sens_review2_A2a_later_admission.csv", a2),
                     ("sens_review2_B9_end_of_followup.csv", b9)]:
        pd.DataFrame(rows).to_csv(results_tables(nm), index=False)
        log(f"\n{nm}\n" + pd.DataFrame(rows).to_string())
    with open(log_path("12_descriptives.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["combine"]:
        allr = pd.concat([pd.read_csv(results_tables(f"sens_review2_{s}.csv")) for s in SCENARIOS], ignore_index=True)
        allr.to_csv(results_tables("sensitivity_review2.csv"), index=False)
        print(allr.to_string())
    elif args == ["descriptives"]:
        descriptives()
    else:
        for a in args:
            run_scenario(a)
