"""
06_primary_analysis.py
------------------------
REBUILT 2026-08-31 (REVISION 3 + composite-outcome addendum + finalized
statistical analysis plan). Runs the full combined-weight analysis for
all 3 comparisons x 2 outcomes at the PRIMARY grace period (48h).

Per (comparison, outcome):
  1. Load the CCW clone dataset from 05 (already has ipcw_weight).
  2. Fit baseline-confounding IPTW (propensity for ACEI/ARB vs comparator
     on baseline covariates), stabilized, trimmed 1st/99th pctile.
  3. COMBINED WEIGHT = baseline IPTW x IPCW (open item from CLAUDE.md
     REVISION 3 resolved here as a product of the two, then the COMBINED
     weight is trimmed at 1st/99th pctile again -- trimming the final
     analytic weight rather than each factor separately).
  4. SMD covariate balance (unweighted vs combined-weighted).
  5. Weighted Cox model, robust SE clustered on subject_id
     (lifelines CoxPHFitter cluster_col, confirmed supported this env).
  6. Patient-clustered bootstrap (1000 reps, seed 42).
  7. E-values for the point estimate and the CI bound closer to the null.
  8. Love plot, weighted KM plot.
Also builds one unweighted Table 1 per comparison (baseline
characteristics by resolved arm, from the mortality-outcome CCW dataset).

Multiplicity (CLAUDE.md, finalized statistical analysis plan): mortality
and AFS are SEPARATE families, each Bonferroni-corrected across the 3
comparator arms (0.05/3=0.017) independently -- flagged in the output,
not auto-suppressed.

Output:
  results/tables/balance_table_<comparison>_<outcome>.csv
  results/tables/primary_results_<comparison>_<outcome>.csv
  results/tables/primary_results_ALL.csv (combined, with outcome column)
  results/tables/table1_<comparison>.csv
  results/figures/love_plot_<comparison>_<outcome>.png
  results/figures/weighted_km_<comparison>_<outcome>.png
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter, KaplanMeierFitter

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from importlib import import_module
ccw_mod = import_module("05_ccw_clone_censor")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

CONFOUNDERS_BASE = ["age_at_admission", "gender_male", "mi", "chf", "cerebrovascular",
                     "chronic_pulmonary", "renal", "sepsis", "charlson_comorbidity_index",
                     "baseline_creatinine", "creatinine_unknown", "baseline_sbp",
                     "baseline_dbp", "bp_unknown", "emergency_admission", "icu_stay_flag"]

BONFERRONI_ALPHA = 0.05 / 3


def smd(x_t, x_c, w_t=None, w_c=None):
    if w_t is None:
        m_t, m_c = x_t.mean(), x_c.mean()
        s_t, s_c = x_t.std(), x_c.std()
    else:
        m_t = np.average(x_t, weights=w_t)
        m_c = np.average(x_c, weights=w_c)
        s_t = np.sqrt(np.average((x_t - m_t) ** 2, weights=w_t))
        s_c = np.sqrt(np.average((x_c - m_c) ** 2, weights=w_c))
    pooled_sd = np.sqrt((s_t ** 2 + s_c ** 2) / 2)
    return (m_t - m_c) / pooled_sd if pooled_sd > 0 else 0.0


def e_value(estimate):
    """VanderWeele & Ding E-value for a risk-ratio-scale estimate
    (Cox HR treated as an RR approximation, standard practice for
    time-to-event estimates without very high cumulative incidence)."""
    rr = estimate if estimate >= 1 else 1 / estimate
    return rr + np.sqrt(rr * (rr - 1))


def load_weighted(comparison, outcome_name):
    """CCW clone dataset with treated flag, ps, iptw and combined_weight.
    Single source of truth for the analytic weight -- run_analysis uses it,
    and the manuscript figure builders import it (refactor 2026-10-01, no
    numerical change)."""
    df = pd.read_parquet(data(f"ccw_{comparison}_{outcome_name}.parquet"))
    df["treated"] = (df["clone_arm"] == "ACEI_ARB").astype(int)
    confounders = [c for c in CONFOUNDERS_BASE if c in df.columns]
    race_cols = [c for c in df.columns if c.startswith("race_") and c != "race_simplified"]
    confounders += race_cols
    return add_combined_weight(df, confounders), confounders


def add_combined_weight(df, confounders):
    """Adds ps, iptw, combined_weight_raw and combined_weight to a CCW clone
    dataset that already carries ipcw_weight. `confounders` is the IPTW
    model's covariate list (full list in the primary analysis; subsets in the
    09 confounding decomposition)."""
    # --- Baseline-confounding IPTW (open item, resolved 2026-08-31) ---
    # DIAGNOSTIC FINDING that motivated this: fitting the propensity model
    # UNWEIGHTED then simply multiplying by ipcw_weight destroys IPTW's
    # own balance -- IPTW alone gives perfect balance (0/20 imbalanced on
    # a test case), but IPCW alone is badly imbalanced (12/20, since it
    # targets informative censoring, not treatment balance, and has much
    # higher variance: IPCW weight std ~7.5 vs IPTW's ~0.4 on the same
    # data). The naive product inherits most of IPCW's imbalance.
    # FIX: fit the propensity model WEIGHTED BY ipcw_weight, so the
    # propensity score already targets the censoring-corrected
    # pseudo-population the combined weight will represent, instead of
    # being blind to it. This is standard practice for combining
    # sequential weights (fit each stage's model on the population implied
    # by the prior stage's weight) and empirically resolved most of the
    # imbalance (test case: 8/20 -> 1/20). Documented in
    # notes/2026-08-31_ccw_rebuild_results.md.
    X = df[confounders].fillna(df[confounders].median()).values
    y = df["treated"].values
    ps_model = LogisticRegression(max_iter=2000)
    ps_model.fit(X, y, sample_weight=df["ipcw_weight"].values)
    df["ps"] = ps_model.predict_proba(X)[:, 1]
    marginal_p = np.average(df["treated"], weights=df["ipcw_weight"])
    df["iptw"] = np.where(df["treated"] == 1, marginal_p / df["ps"], (1 - marginal_p) / (1 - df["ps"]))

    # --- Combined weight: baseline IPTW x IPCW, trimmed ONCE as the
    # final analytic weight (not double-trimmed per factor) ---
    df["combined_weight_raw"] = df["iptw"] * df["ipcw_weight"]
    lo_c, hi_c = df["combined_weight_raw"].quantile([0.01, 0.99])
    df["combined_weight"] = df["combined_weight_raw"].clip(lower=lo_c, upper=hi_c)
    return df


RISK_HORIZON_DAYS = 365   # absolute-risk horizon (added 2026-11-10)


def weighted_km_risk(time, event, w, horizon=RISK_HORIZON_DAYS):
    """1 - weighted Kaplan-Meier survival at `horizon` days."""
    order = np.argsort(time, kind="mergesort")
    t, e, w = np.asarray(time)[order], np.asarray(event)[order], np.asarray(w, dtype=float)[order]
    at_risk = np.cumsum(w[::-1])[::-1]
    s = 1.0
    uniq = np.unique(t[(e == 1) & (t <= horizon)])
    for u in uniq:
        idx = np.searchsorted(t, u, side="left")
        d = w[(t == u) & (e == 1)].sum()
        s *= 1 - d / at_risk[idx]
    return 1 - s


def arm_risks(df, weight_col):
    w = df[weight_col].values if weight_col else np.ones(len(df))
    out = {}
    for arm in (1, 0):
        m = (df["treated"] == arm).values
        out[arm] = weighted_km_risk(df["time"].values[m], df["event"].values[m], w[m])
    return out[1], out[0]


def cox_hr_bootstrap(df, weight_col="combined_weight"):
    """Weighted Cox HR (robust SE clustered on subject_id) + patient-level
    bootstrap CI (N_BOOT reps, SEED). weight_col=None fits an unweighted
    model. Returns dict(HR, cox_p, ci_lo, ci_hi, n_boot, n_fail) plus, since
    2026-11-10, the weighted 1-year risk per arm and the risk difference
    (ACEI/ARB minus comparator) with bootstrap CI from the SAME resamples."""
    cols = ["treated", "event", "time", "subject_id"] + ([weight_col] if weight_col else [])
    cph = CoxPHFitter()
    cph.fit(df[cols], duration_col="time", event_col="event", weights_col=weight_col,
            cluster_col="subject_id", robust=True, formula="treated")
    point_hr = cph.hazard_ratios_["treated"]
    cox_p = cph.summary.loc["treated", "p"]
    risk_t, risk_c = arm_risks(df, weight_col)

    rng = np.random.default_rng(SEED)
    patient_ids = df["subject_id"].unique()
    boot_hrs, boot_rds, boot_failures = [], [], []
    for b in range(N_BOOT):
        boot_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        counts = pd.Series(boot_ids).value_counts()
        row_mult = df["subject_id"].map(counts).fillna(0).astype(int)
        rep_df = df.loc[df.index.repeat(row_mult)].reset_index(drop=True)
        try:
            cph_b = CoxPHFitter()
            cph_b.fit(rep_df[[c for c in cols if c != "subject_id"]],
                      duration_col="time", event_col="event", weights_col=weight_col,
                      robust=True, formula="treated")
            boot_hrs.append(cph_b.hazard_ratios_["treated"])
            rt, rc = arm_risks(rep_df, weight_col)
            boot_rds.append(rt - rc)
        except Exception as e:
            boot_failures.append(str(e))
    boot_hrs = np.array(boot_hrs)
    ci_lo, ci_hi = np.percentile(boot_hrs, [2.5, 97.5])
    rd_lo, rd_hi = np.percentile(boot_rds, [2.5, 97.5])
    return {"HR": point_hr, "cox_p": cox_p, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_boot": len(boot_hrs), "n_fail": len(boot_failures),
            "risk_treated_1y": risk_t, "risk_control_1y": risk_c,
            "rd_1y": risk_t - risk_c, "rd_ci_lo": rd_lo, "rd_ci_hi": rd_hi}


def ph_diagnostics(df, weight_col="combined_weight", split_day=30):
    """Proportional-hazards checks (added 2026-11-10): rank-transformed
    Schoenfeld-residual test on the weighted model, and piecewise HRs for
    days 0-split_day and split_day-365+ (left-truncated at split_day)."""
    from lifelines.statistics import proportional_hazard_test
    cols = ["treated", "event", "time", weight_col]
    cph = CoxPHFitter().fit(df[cols], duration_col="time", event_col="event",
                            weights_col=weight_col, robust=True)
    p_ph = proportional_hazard_test(cph, df[cols], time_transform="rank").summary["p"].iloc[0]
    early = df[cols].copy()
    early["event"] = ((early["time"] <= split_day) & (early["event"] == 1)).astype(int)
    early["time"] = early["time"].clip(upper=split_day)
    hr_early = CoxPHFitter().fit(early, "time", "event", weights_col=weight_col,
                                 robust=True).hazard_ratios_["treated"]
    late = df.loc[df["time"] > split_day, cols].copy()
    late["entry"] = split_day
    hr_late = CoxPHFitter().fit(late, "time", "event", weights_col=weight_col, entry_col="entry",
                                robust=True).hazard_ratios_["treated"]
    return {"ph_test_p": p_ph, f"HR_0_{split_day}d": hr_early, f"HR_{split_day}d_plus": hr_late}


def run_analysis(comparison, outcome_name):
    log(f"\n{'=' * 70}\n{comparison} vs ACEI/ARB -- outcome: {outcome_name}\n{'=' * 70}")

    df, confounders = load_weighted(comparison, outcome_name)
    log(f"N={len(df)}, treated (ACEI/ARB clone)={df.treated.sum()}, "
        f"control ({comparison} clone)={(1 - df.treated).sum()}")
    log(f"Events: treated={df.loc[df.treated==1,'event'].sum()}, "
        f"control={df.loc[df.treated==0,'event'].sum()}")

    treated_mask = df.treated == 1
    balance_rows = []
    for col in confounders:
        x_t, x_c = df.loc[treated_mask, col].values, df.loc[~treated_mask, col].values
        w_t, w_c = df.loc[treated_mask, "combined_weight"].values, df.loc[~treated_mask, "combined_weight"].values
        balance_rows.append({"covariate": col, "smd_unweighted": smd(x_t, x_c),
                              "smd_weighted": smd(x_t, x_c, w_t, w_c)})
    balance_df = pd.DataFrame(balance_rows)
    balance_df["balanced_after_weighting"] = balance_df["smd_weighted"].abs() < 0.10
    balance_df.to_csv(results_tables(f"balance_table_{comparison}_{outcome_name}.csv"), index=False)
    n_imbalanced = (~balance_df.balanced_after_weighting).sum()
    log(f"Covariates imbalanced after combined weighting (|SMD|>=0.10): {n_imbalanced} / {len(confounders)}")

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(balance_df))
    ax.scatter(balance_df.smd_unweighted.abs(), y_pos, label="Unweighted", color="grey", marker="o")
    ax.scatter(balance_df.smd_weighted.abs(), y_pos, label="Combined-weighted (IPTW x IPCW)", color="steelblue", marker="s")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df.covariate)
    ax.axvline(0.10, linestyle="--", color="red", label="SMD=0.10 threshold")
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title(f"Covariate balance: ACEI/ARB vs {comparison} ({outcome_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_figures(f"love_plot_{comparison}_{outcome_name}.png"), dpi=300)
    plt.close(fig)

    fit = cox_hr_bootstrap(df)
    point_hr, cox_p, ci_lo, ci_hi = fit["HR"], fit["cox_p"], fit["ci_lo"], fit["ci_hi"]
    log(f"Point estimate HR = {point_hr:.3f}, p={cox_p:.3f}")
    log(f"Bootstrap ({fit['n_boot']}/{N_BOOT} successful, {fit['n_fail']} failed): "
        f"HR={point_hr:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    log(f"Weighted 1-year risk: ACEI/ARB={100*fit['risk_treated_1y']:.1f}%, {comparison}="
        f"{100*fit['risk_control_1y']:.1f}%, RD={100*fit['rd_1y']:.1f} pp "
        f"[{100*fit['rd_ci_lo']:.1f}, {100*fit['rd_ci_hi']:.1f}]")
    phd = ph_diagnostics(df)
    log(f"PH check: Schoenfeld (rank) p={phd['ph_test_p']:.3f}; HR days 0-30={phd['HR_0_30d']:.2f}, "
        f"days 30+={phd['HR_30d_plus']:.2f}")
    assert fit["n_boot"] >= 0.95 * N_BOOT, (
        f"Too many bootstrap failures for {comparison}/{outcome_name} "
        f"({fit['n_fail']}/{N_BOOT}) -- investigate."
    )

    ci_bound_near_null = ci_lo if point_hr >= 1 else ci_hi
    # If the CI crosses 1, the bound "closer to null" IS 1 -> E-value is
    # undefined/1 for that bound (no confounding needed to explain a
    # non-significant result away); report as NaN rather than a
    # misleadingly precise number.
    ci_crosses_null = ci_lo <= 1 <= ci_hi
    ev_point = e_value(point_hr)
    ev_ci = np.nan if ci_crosses_null else e_value(ci_bound_near_null)
    log(f"E-value: point={ev_point:.2f}, CI-bound-near-null={ev_ci if not np.isnan(ev_ci) else 'NA (CI crosses 1)'}")

    passes_bonferroni = cox_p < BONFERRONI_ALPHA

    fig, ax = plt.subplots(figsize=(7, 6))
    for arm_val, label, color in [(1, "ACEI/ARB", "#2166AC"), (0, comparison, "#D6604D")]:
        sub = df[df.treated == arm_val]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], sub["event"], weights=sub["combined_weight"], label=label)
        kmf.plot_survival_function(ax=ax, color=color)
    ax.set_xlabel("Days from time zero (admission)")
    ax.set_ylabel("Combined-weighted survival probability")
    ax.set_title(f"ACEI/ARB vs {comparison} ({outcome_name}): HR={point_hr:.2f}, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    fig.tight_layout()
    fig.savefig(results_figures(f"weighted_km_{comparison}_{outcome_name}.png"), dpi=300)
    plt.close(fig)

    result_row = {
        "comparison": f"ACEI_ARB_vs_{comparison}", "outcome": outcome_name,
        "n_treated": int(df.treated.sum()), "n_control": int((1 - df.treated).sum()),
        "events_treated": int(df.loc[treated_mask, "event"].sum()),
        "events_control": int(df.loc[~treated_mask, "event"].sum()),
        "HR": point_hr, "HR_CI_lower": ci_lo, "HR_CI_upper": ci_hi, "cox_p_value": cox_p,
        "bonferroni_alpha": BONFERRONI_ALPHA, "passes_bonferroni_within_family": passes_bonferroni,
        "e_value_point": ev_point, "e_value_ci_bound": ev_ci,
        "n_boot_successful": fit["n_boot"], "n_covariates_imbalanced": int(n_imbalanced),
        "risk_treated_1y": fit["risk_treated_1y"], "risk_control_1y": fit["risk_control_1y"],
        "rd_1y": fit["rd_1y"], "rd_1y_CI_lower": fit["rd_ci_lo"], "rd_1y_CI_upper": fit["rd_ci_hi"],
        **phd,
    }
    pd.DataFrame([result_row]).to_csv(results_tables(f"primary_results_{comparison}_{outcome_name}.csv"), index=False)
    return result_row, df


def build_table1(comparison, mortality_df):
    rows = []
    for col in CONFOUNDERS_BASE:
        if col not in mortality_df.columns:
            continue
        t = mortality_df.loc[mortality_df.treated == 1, col]
        c = mortality_df.loc[mortality_df.treated == 0, col]
        if set(t.dropna().unique()) <= {0, 1}:
            rows.append({"characteristic": col, "ACEI_ARB": f"{100*t.mean():.1f}%", comparison: f"{100*c.mean():.1f}%"})
        else:
            rows.append({"characteristic": col,
                         "ACEI_ARB": f"{t.mean():.1f} ({t.std():.1f})",
                         comparison: f"{c.mean():.1f} ({c.std():.1f})"})
    rows.append({"characteristic": "N", "ACEI_ARB": int((mortality_df.treated == 1).sum()),
                 comparison: int((mortality_df.treated == 0).sum())})
    pd.DataFrame(rows).to_csv(results_tables(f"table1_{comparison}.csv"), index=False)


if __name__ == "__main__":
    # BUG FIX 2026-08-31: this block used to run at MODULE LEVEL (no
    # __main__ guard), so every `import_module("06_primary_analysis")`
    # (done by script 07, and by ad hoc diagnostics) silently re-ran the
    # full 6-comparison pipeline as an import side effect -- this is why
    # script 07 took far longer than its own sensitivity loop alone
    # should have. Fixed so the module can be imported cheaply for its
    # functions/constants (CONFOUNDERS_BASE, smd, e_value, run_analysis).
    # Outcome hierarchy REVISION (2026-08-31): AFS is now PRIMARY, run
    # first per comparison; mortality and amputation-alone are SECONDARY.
    all_results = []
    for comparison in ccw_mod.COMPARATORS:
        afs_result, afs_df = run_analysis(comparison, "afs")
        all_results.append(afs_result)
        mortality_result, mortality_df = run_analysis(comparison, "mortality")
        all_results.append(mortality_result)
        amputation_result, amputation_df = run_analysis(comparison, "amputation")
        all_results.append(amputation_result)
        build_table1(comparison, mortality_df)
        log(f"Saved: results/tables/table1_{comparison}.csv")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(results_tables("primary_results_ALL.csv"), index=False)
    log(f"\n{'=' * 70}\nSummary across all comparisons x outcomes:\n{summary_df.to_string()}")

    with open(log_path("06_primary_analysis.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
