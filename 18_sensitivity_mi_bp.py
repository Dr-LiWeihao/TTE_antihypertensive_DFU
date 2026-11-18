"""
18_sensitivity_mi_bp.py
-----------------------
NEW 2026-11-13. Multiple-imputation sensitivity analysis for missing outpatient (omr) blood pressure
(review v3, R2-m4). Pre-specified BEFORE running in notes/2026-11-13_mi_bp_figures_length_prespec.md (M1).

The primary analysis median-imputes baseline SBP/DBP and adds a missing indicator (bp_unknown). Here the
missing values are multiply imputed (m = 20, chained equations, outcome information in the imputation model),
bp_unknown is dropped from both weight models, the primary CCW/IPCW/IPTW/Cox specification is run in each
imputed dataset, and log HRs are pooled with Rubin's rules using the patient-clustered robust SE.

Outputs: results/tables/sensitivity_mi_bp.csv, results/tables/sensitivity_mi_bp_imputations.csv,
         logs/18_sensitivity_mi_bp.log
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, NelsonAalenFitter
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from importlib import import_module
ccw_mod = import_module("05_ccw_clone_censor")
primary = import_module("06_primary_analysis")

M = 20
G = GRACE_HOURS_PRIMARY
log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))


def build_imputations():
    cov = pd.read_parquet(data("covariates.parquet"))
    out = pd.read_parquet(data("outcomes.parquet"))
    drugs = pd.read_parquet(data("drug_exposure.parquet"))
    r3 = pd.read_parquet(data("review3_covariates.parquet"))
    d = cov.merge(out[["hadm_id", "mortality_event", "mortality_days_from_t0", "afs_eligible",
                       "amputation_only_event", "amputation_only_days_from_t0"]], on="hadm_id", how="left")
    d = d.merge(drugs[["hadm_id"] + [f"{c}_{G}h" for c in ["aceiarb", "ccb", "betablocker", "diuretic"]]],
                on="hadm_id", how="left")
    d = d.merge(r3[["hadm_id", "ed_sbp", "ed_dbp"]], on="hadm_id", how="left")
    assert len(d) == len(cov) == d.hadm_id.nunique()

    miss = d["bp_unknown"] == 1
    d.loc[miss, ["baseline_sbp", "baseline_dbp"]] = np.nan          # undo the median fill
    obs_range = {c: (d[c].min(), d[c].max()) for c in ["baseline_sbp", "baseline_dbp", "ed_sbp", "ed_dbp"]}

    # Outcome information (White & Royston 2009): event indicator + Nelson-Aalen cumulative hazard
    na = NelsonAalenFitter().fit(d["mortality_days_from_t0"], d["mortality_event"])
    d["H_mort"] = na.cumulative_hazard_at_times(d["mortality_days_from_t0"]).values
    el = d["afs_eligible"].astype(bool)
    na2 = NelsonAalenFitter().fit(d.loc[el, "amputation_only_days_from_t0"], d.loc[el, "amputation_only_event"])
    d["amp_ev"] = np.where(el, d["amputation_only_event"].fillna(0), 1).astype(float)
    d["H_amp"] = np.where(el, na2.cumulative_hazard_at_times(d["amputation_only_days_from_t0"].fillna(0)).values,
                          float(na2.cumulative_hazard_at_times([0]).iloc[0]))

    d["gender_male"] = (d["gender"] == "M").astype(int)
    race = pd.get_dummies(d["race_simplified"], prefix="r", drop_first=True).astype(int)
    pred = ["age_at_admission", "gender_male", "emergency_admission", "icu_stay_flag", "mi", "chf", "cerebrovascular",
            "chronic_pulmonary", "renal", "sepsis", "charlson_comorbidity_index", "baseline_creatinine",
            "creatinine_unknown"] + [f"{c}_{G}h" for c in ["aceiarb", "ccb", "betablocker", "diuretic"]] + \
           ["mortality_event", "H_mort", "amp_ev", "H_amp"]
    targets = ["baseline_sbp", "baseline_dbp", "ed_sbp", "ed_dbp"]
    X = pd.concat([d[targets], d[pred].astype(float), race], axis=1)
    log(f"Admissions {len(d):,}; outpatient BP missing {miss.sum():,} ({100 * miss.mean():.1f}%); "
        f"ED BP missing {d.ed_sbp.isna().sum():,}; imputation-model columns {X.shape[1]}")

    imps = []
    for k in range(M):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=20, random_state=SEED + k)
        Z = pd.DataFrame(imp.fit_transform(X.values), columns=X.columns, index=X.index)
        e = pd.DataFrame({"hadm_id": d["hadm_id"].values})
        for c, nm in [("baseline_sbp", "mi_sbp"), ("baseline_dbp", "mi_dbp")]:
            lo, hi = obs_range[c]
            e[nm] = Z[c].clip(lo, hi).values
        log(f"  imputation {k + 1:2d}: imputed SBP mean {e.loc[miss.values, 'mi_sbp'].mean():.1f} "
            f"(observed {d.loc[~miss, 'baseline_sbp'].mean():.1f}); DBP {e.loc[miss.values, 'mi_dbp'].mean():.1f} "
            f"(observed {d.loc[~miss, 'baseline_dbp'].mean():.1f})")
        imps.append(e)
    return imps


def swap(cols):
    out = []
    for c in cols:
        if c == "bp_unknown":
            continue
        out.append({"baseline_sbp": "mi_sbp", "baseline_dbp": "mi_dbp"}.get(c, c))
    return out


IPCW_MI = swap(ccw_mod.IPCW_COVARIATES)
CONF_MI = swap(primary.CONFOUNDERS_BASE)


def fit_one(comparison, outcome, extra, ipcw_cov, conf_base):
    df = ccw_mod.build_ccw_dataset(G, comparison, outcome, log=lambda m: None, extra_covariates=extra,
                                   ipcw_covariates=ipcw_cov)
    df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
    conf = [c for c in conf_base if c in df.columns] + \
           [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
    df = primary.add_combined_weight(df, conf)
    cph = CoxPHFitter().fit(df[["treated", "event", "time", "subject_id", "combined_weight"]], duration_col="time",
                            event_col="event", weights_col="combined_weight", cluster_col="subject_id",
                            robust=True, formula="treated")
    t = df.treated == 1
    n_imb = int(sum(abs(primary.smd(df.loc[t, c].values, df.loc[~t, c].values, df.loc[t, "combined_weight"].values,
                                    df.loc[~t, "combined_weight"].values)) >= 0.10 for c in conf))
    return {"logHR": float(cph.params_["treated"]), "se": float(cph.summary.loc["treated", "se(coef)"]),
            "n_imbalanced": n_imb, "n_cov": len(conf), "n_treated": int(t.sum()), "n_control": int((~t).sum()),
            "events_treated": int(df.loc[t, "event"].sum()), "events_control": int(df.loc[~t, "event"].sum())}


def rubin(q, u):
    q, u = np.asarray(q), np.asarray(u)
    m = len(q)
    qbar, ubar, b = q.mean(), u.mean(), q.var(ddof=1)
    tvar = ubar + (1 + 1 / m) * b
    r = (1 + 1 / m) * b / ubar
    df = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else np.inf
    fmi = (r + 2 / (df + 3)) / (r + 1) if np.isfinite(df) else 0.0
    se = np.sqrt(tvar)
    tq = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(qbar / se), df)
    return qbar, se, qbar - tq * se, qbar + tq * se, p, df, fmi


if __name__ == "__main__":
    imps = build_imputations()
    per, rows = [], []
    for comparison in ccw_mod.COMPARATORS:
        for outcome in ccw_mod.OUTCOMES:
            ref = fit_one(comparison, outcome, None, None, list(primary.CONFOUNDERS_BASE))
            fits = []
            for k, e in enumerate(imps):
                f = fit_one(comparison, outcome, e, IPCW_MI, CONF_MI)
                f.update({"comparison": comparison, "outcome": outcome, "imputation": k + 1})
                fits.append(f)
                per.append(f)
            qbar, se, lo, hi, p, df_, fmi = rubin([f["logHR"] for f in fits], [f["se"] ** 2 for f in fits])
            row = {"comparison": comparison, "outcome": outcome, "m": M,
                   "HR_mi": np.exp(qbar), "HR_mi_CI_lower": np.exp(lo), "HR_mi_CI_upper": np.exp(hi), "p_mi": p,
                   "rubin_df": df_, "fmi": fmi,
                   "max_n_imbalanced": max(f["n_imbalanced"] for f in fits), "n_cov": fits[0]["n_cov"],
                   "HR_primary": np.exp(ref["logHR"]),
                   "HR_primary_robust_CI_lower": np.exp(ref["logHR"] - 1.959964 * ref["se"]),
                   "HR_primary_robust_CI_upper": np.exp(ref["logHR"] + 1.959964 * ref["se"]),
                   "n_treated_primary": ref["n_treated"], "n_control_primary": ref["n_control"]}
            rows.append(row)
            log(f"{comparison:12s} {outcome:10s} MI HR {row['HR_mi']:.3f} [{row['HR_mi_CI_lower']:.3f}, "
                f"{row['HR_mi_CI_upper']:.3f}] p={p:.3f} FMI={fmi:.3f} max imbalanced {row['max_n_imbalanced']}/"
                f"{row['n_cov']} | primary {row['HR_primary']:.3f} [{row['HR_primary_robust_CI_lower']:.3f}, "
                f"{row['HR_primary_robust_CI_upper']:.3f}] (robust SE)")
    pd.DataFrame(rows).to_csv(results_tables("sensitivity_mi_bp.csv"), index=False)
    pd.DataFrame(per).to_csv(results_tables("sensitivity_mi_bp_imputations.csv"), index=False)
    with open(log_path("18_sensitivity_mi_bp.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))
