"""
build_bmc_tables.py
-------------------
Main-text Tables 2-3 and Additional file 1 Tables S1-S8 for the BMC Cardiovascular
Disorders submission, as clean CSVs. Adapted 2026-11-10 from
Version_2_JEI/scripts/build_jei_tables.py. Reads only analysis outputs
(results/tables/*.csv, data/ccw_*.parquet) and scripts/config.py code lists;
no statistic is typed in by hand.

Table 1 (protocol, text only) is maintained by hand in tables/Table1_protocol.csv.
New since Version_3: Table 3 has a 1-year risk-difference column; Table S6 (absolute
risks and proportional-hazards checks), Table S7 (comparator-definition sensitivity,
script 10) and Table S8 (drug and amputation code lists). Since 2026-11-10 also Tables S9-S12
(script 12) and Table S13 plus the beta-blocker-type rows of Table S7 and the any-amputation
rows of Table S9 (script 15).

Output: manuscript/Version_4_BMC/tables/
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg

RES = os.path.join(cfg.RES_DIR, "tables")
DATA = cfg.DATA_DIR
OUT = os.path.join(cfg.OUT_DIR, "manuscript_tables")
os.makedirs(OUT, exist_ok=True)

COMPS = ["CCB", "betablocker", "diuretic"]
COMP_LABEL = {"CCB": "CCB", "betablocker": "Beta-blocker", "diuretic": "Diuretic",
              "CCB_DHP": "Dihydropyridine CCB", "diuretic_thiazide": "Thiazide-type diuretic, no loop diuretic",
              "BB_vasodilating": "Vasodilating beta-blocker only", "BB_nonvasodilating": "Non-vasodilating beta-blocker only"}
OUTCOMES = ["afs", "mortality", "amputation"]
OUTCOME_LABEL = {"afs": "Amputation-free survival (primary)", "mortality": "All-cause mortality",
                 "amputation": "Major amputation"}
COV_LABEL = {
    "age_at_admission": "Age, years", "gender_male": "Male sex", "mi": "Myocardial infarction",
    "chf": "Congestive heart failure", "cerebrovascular": "Cerebrovascular disease",
    "chronic_pulmonary": "Chronic pulmonary disease", "renal": "Renal disease", "sepsis": "Sepsis",
    "charlson_comorbidity_index": "Charlson Comorbidity Index",
    "baseline_creatinine": "Baseline creatinine, mg/dL", "creatinine_unknown": "Baseline creatinine missing",
    "baseline_sbp": "Baseline systolic BP, mmHg", "baseline_dbp": "Baseline diastolic BP, mmHg",
    "bp_unknown": "Baseline BP missing", "emergency_admission": "Emergency or urgent admission",
    "icu_stay_flag": "ICU stay during admission", "race_Black": "Race: Black", "race_Hispanic": "Race: Hispanic",
    "race_Other/Unknown": "Race: other or unknown", "race_White": "Race: White", "race_Asian": "Race: Asian",
}
CONTINUOUS = {"age_at_admission", "charlson_comorbidity_index", "baseline_creatinine", "baseline_sbp", "baseline_dbp"}


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_hr(hr, lo, hi):
    return f"{hr:.2f} ({lo:.2f}–{hi:.2f})"


def fmt_rd(rd, lo, hi):
    m = lambda x: f"{100 * x:.1f}".replace("-", "−")  # typographic minus
    return f"{m(rd)} ({m(lo)} to {m(hi)})"


# --- Table 2: baseline characteristics (unweighted, retained clones) ------
cols = {}
acei_ids = None
for c in COMPS:
    d = pd.read_parquet(os.path.join(DATA, f"ccw_{c}_mortality.parquet"))
    d["race_Asian"] = (d["race_simplified"] == "Asian").astype(int)
    for arm_name, sub in [("ACEI/ARB", d[d.clone_arm == "ACEI_ARB"]), (COMP_LABEL[c], d[d.clone_arm != "ACEI_ARB"])]:
        if arm_name == "ACEI/ARB":
            ids = tuple(sorted(sub.hadm_id))
            assert acei_ids is None or ids == acei_ids, "ACEI/ARB retained set differs across comparisons"
            acei_ids = ids
        col = {"Admissions, n": f"{len(sub)}", "Patients, n": f"{sub.subject_id.nunique()}"}
        for v in ["age_at_admission", "gender_male", "race_White", "race_Black", "race_Hispanic", "race_Asian",
                  "race_Other/Unknown", "emergency_admission", "icu_stay_flag", "mi", "chf", "cerebrovascular",
                  "chronic_pulmonary", "renal", "sepsis", "charlson_comorbidity_index", "baseline_creatinine",
                  "creatinine_unknown", "baseline_sbp", "baseline_dbp", "bp_unknown"]:
            x = sub[v].astype(float)
            col[COV_LABEL[v] + (", mean (SD)" if v in CONTINUOUS else ", %")] = (
                f"{x.mean():.1f} ({x.std():.1f})" if v in CONTINUOUS else f"{100 * x.mean():.1f}")
        cols[arm_name] = col
t2 = pd.DataFrame(cols)[["ACEI/ARB", "CCB", "Beta-blocker", "Diuretic"]]
t2.index.name = "Characteristic"
t2.to_csv(os.path.join(OUT, "Table2_baseline.csv"))

# --- Table 3: main results (+ 1-year risk difference) ----------------------
res = pd.read_csv(os.path.join(RES, "primary_results_ALL.csv"))
res["comp"] = res["comparison"].str.replace("ACEI_ARB_vs_", "", regex=False)
# Major amputation absolute risk: weighted Aalen-Johansen CIF with death as competing event (script 13)
cif = pd.read_csv(os.path.join(RES, "amputation_cuminc.csv")).set_index("comparison")
rows = []
for o in OUTCOMES:
    for c in COMPS:
        r = res[(res.outcome == o) & (res.comp == c)].iloc[0]
        rows.append({
            "Outcome": OUTCOME_LABEL[o], "Comparator": COMP_LABEL[c],
            "ACEI/ARB, events/n": f"{int(r.events_treated)}/{int(r.n_treated)}",
            "Comparator, events/n": f"{int(r.events_control)}/{int(r.n_control)}",
            "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper),
            "p": fmt_p(r.cox_p_value),
            "1-year RD, % points (95% CI)": fmt_rd(r.rd_1y, r.rd_1y_CI_lower, r.rd_1y_CI_upper)
            if o != "amputation" else fmt_rd(cif.loc[c, "rd_1y"], cif.loc[c, "rd_1y_CI_lower"],
                                             cif.loc[c, "rd_1y_CI_upper"]),
        })
pd.DataFrame(rows).to_csv(os.path.join(OUT, "Table3_results.csv"), index=False)

# --- Table S2: covariate balance -------------------------------------------
s1 = []
for o in OUTCOMES:
    for c in COMPS:
        b = pd.read_csv(os.path.join(RES, f"balance_table_{c}_{o}.csv"))
        for _, r in b.iterrows():
            s1.append({"Outcome": OUTCOME_LABEL[o], "Comparator": COMP_LABEL[c],
                       "Covariate": COV_LABEL.get(r.covariate, r.covariate),
                       "SMD unweighted": f"{r.smd_unweighted:.3f}", "SMD weighted": f"{r.smd_weighted:.3f}"})
pd.DataFrame(s1).to_csv(os.path.join(OUT, "TableS2_balance.csv"), index=False)

# --- Table S3: grace-period sensitivity -------------------------------------
sens = pd.read_csv(os.path.join(RES, "sensitivity_grace_period.csv"))
s2 = []
for o in OUTCOMES:
    for c in COMPS:
        for g in [24, 48, 72]:
            r = sens[(sens.outcome == o) & (sens.comparison == c) & (sens.grace_hours == g)].iloc[0]
            s2.append({"Outcome": OUTCOME_LABEL[o], "Comparator": COMP_LABEL[c],
                       "Grace period": f"{g} h" + (" (primary)" if g == 48 else ""),
                       "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper), "p": fmt_p(r.cox_p_value)})
pd.DataFrame(s2).to_csv(os.path.join(OUT, "TableS3_grace_period.csv"), index=False)

# --- Table S5: E-values ------------------------------------------------------
# REVISED 2026-11-13 (review v3, R2-m1): the outcomes are common (AFS ~31%, mortality ~20%), so the HR is
# converted to the risk-ratio scale with VanderWeele's common-outcome approximation
# RR = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)) before computing the E-value (VanderWeele & Ding 2017).
import numpy as np


def hr_to_rr_common(hr):
    return (1 - 0.5 ** np.sqrt(hr)) / (1 - 0.5 ** np.sqrt(1 / hr))


def evalue_rr(rr):
    rr = rr if rr >= 1 else 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


s3 = []
for o in OUTCOMES:
    for c in COMPS:
        r = res[(res.outcome == o) & (res.comp == c)].iloc[0]
        rr = hr_to_rr_common(r.HR)
        if r.HR_CI_lower <= 1 <= r.HR_CI_upper:
            e_ci = "1.00 (CI includes 1)"
        else:
            lim = r.HR_CI_lower if r.HR > 1 else r.HR_CI_upper
            e_ci = f"{evalue_rr(hr_to_rr_common(lim)):.2f}"
        s3.append({"Outcome": OUTCOME_LABEL[o], "Comparator": COMP_LABEL[c],
                   "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper),
                   "Approximate RR": f"{rr:.2f}",
                   "E-value, point estimate": f"{evalue_rr(rr):.2f}",
                   "E-value, CI bound": e_ci})
pd.DataFrame(s3).to_csv(os.path.join(OUT, "TableS5_evalues.csv"), index=False)

# --- Table S4: extreme-weight handling --------------------------------------
ew = pd.read_csv(os.path.join(RES, "extreme_weight_sensitivity.csv"))
RULE = {"trim_1_99": "Trimmed at 1st/99th percentile (primary)", "trim_2p5_97p5": "Trimmed at 2.5th/97.5th percentile",
        "exclude_top1pct": "Admissions above 99th percentile excluded"}
s4 = []
for _, r in ew.iterrows():
    s4.append({"Outcome": OUTCOME_LABEL[r.outcome], "Comparator": COMP_LABEL[r.comparison.replace("ACEI_ARB_vs_", "")],
               "Weight handling": RULE.get(r.weight_rule, r.weight_rule),
               "Clones analysed": f"{int(r.n_analyzed)}",
               "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper), "p": fmt_p(r.cox_p_value),
               "Significant after Bonferroni": "Yes" if r.passes_bonferroni_within_family else "No"})
pd.DataFrame(s4).to_csv(os.path.join(OUT, "TableS4_extreme_weights.csv"), index=False)

# --- Table S1: crude-to-adjusted decomposition (09, exploratory) ----------
dec = pd.read_csv(os.path.join(RES, "confounding_decomposition.csv"))
STEP = {"M0_crude": "Crude (unweighted)", "M1_demographics": "Demographics (age, sex, race)",
        "M2a_demographics_cardiorenal": "Demographics + cardiorenal factors",
        "M2b_demographics_other": "Demographics + other factors", "M3_full": "Fully adjusted (primary analysis)"}
s5 = []
for o in OUTCOMES:
    for c in COMPS:
        for m in STEP:
            r = dec[(dec.outcome == o) & (dec.comparison == c) & (dec.model == m)].iloc[0]
            s5.append({"Outcome": OUTCOME_LABEL[o], "Comparator": COMP_LABEL[c], "Adjustment set": STEP[m],
                       "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper), "p": fmt_p(r.cox_p_value),
                       "CI method": "Bootstrap" if r.ci_method == "bootstrap" else "Robust"})
pd.DataFrame(s5).to_csv(os.path.join(OUT, "TableS1_decomposition.csv"), index=False)

# --- Table S6: absolute risks and proportional-hazards checks -------------
s6 = []
for o in OUTCOMES:
    for c in COMPS:
        r = res[(res.outcome == o) & (res.comp == c)].iloc[0]
        if o == "amputation":
            k = cif.loc[c]
            rt, rc, rd = k.cif_1y_treated, k.cif_1y_control, fmt_rd(k.rd_1y, k.rd_1y_CI_lower, k.rd_1y_CI_upper)
        else:
            rt, rc, rd = r.risk_treated_1y, r.risk_control_1y, fmt_rd(r.rd_1y, r.rd_1y_CI_lower, r.rd_1y_CI_upper)
        s6.append({"Outcome": OUTCOME_LABEL[o], "Comparator": COMP_LABEL[c],
                   "1-year risk, ACEI/ARB, %": f"{100 * rt:.1f}",
                   "1-year risk, comparator, %": f"{100 * rc:.1f}",
                   "RD, % points (95% CI)": rd,
                   "HR days 0–30": f"{r.HR_0_30d:.2f}", "HR after day 30": f"{r.HR_30d_plus:.2f}",
                   "PH test p": fmt_p(r.ph_test_p)})
pd.DataFrame(s6).to_csv(os.path.join(OUT, "TableS6_absolute_risk_ph.csv"), index=False)

# --- Table S7: comparator-definition sensitivity (script 10) ---------------
cd = pd.read_csv(os.path.join(RES, "sensitivity_comparator_definition.csv"))
s7 = []
for _, r in cd.iterrows():
    s7.append({"Outcome": OUTCOME_LABEL[r.outcome],
               "Comparator strategy": COMP_LABEL[r.comparison.replace("ACEI_ARB_vs_", "")],
               "ACEI/ARB, events/n": f"{int(r.events_treated)}/{int(r.n_treated)}",
               "Comparator, events/n": f"{int(r.events_control)}/{int(r.n_control)}",
               "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper), "p": fmt_p(r.cox_p_value),
               "Covariates with SMD ≥0.10": f"{int(r.n_covariates_imbalanced)}/{int(r.n_covariates)}"})
bbx = pd.read_csv(os.path.join(RES, "bb_pad_exploration.csv"))  # script 15
for _, r in bbx[bbx.task == "E2_subtype"].iterrows():
    s7.append({"Outcome": OUTCOME_LABEL[r.outcome], "Comparator strategy": COMP_LABEL[r.comparison],
               "ACEI/ARB, events/n": f"{int(r.events_treated)}/{int(r.n_treated)}",
               "Comparator, events/n": f"{int(r.events_control)}/{int(r.n_control)}",
               "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper), "p": fmt_p(r.cox_p_value),
               "Covariates with SMD ≥0.10": f"{int(r.n_covariates_imbalanced)}/{int(r.n_covariates)}"})
pd.DataFrame(s7).to_csv(os.path.join(OUT, "TableS7_comparator_definition.csv"), index=False)

# --- Table S8: code lists ---------------------------------------------------
s8 = [
    ("ACEI", ", ".join(cfg.ACEI_DRUGS)),
    ("ARB", ", ".join(cfg.ARB_DRUGS) + " (sacubitril/valsartan included)"),
    ("CCB, dihydropyridine", ", ".join(cfg.CCB_DHP_DRUGS)),
    ("CCB, non-dihydropyridine", ", ".join(cfg.CCB_NONDHP_DRUGS)),
    ("Beta-blocker", ", ".join(cfg.BETABLOCKER_DRUGS)),
    ("Diuretic, thiazide-type", ", ".join(cfg.THIAZIDE_DRUGS)),
    ("Diuretic, loop", ", ".join(cfg.LOOP_DRUGS).replace("ethacrynic", "ethacrynic acid")),
    ("Diuretic, mineralocorticoid-receptor antagonist or potassium-sparing", ", ".join(cfg.MRA_KSPARING_DRUGS)),
    ("Major amputation, ICD-9-CM procedure", ", ".join(f"{c[:2]}.{c[2:]}" for c in cfg.MAJOR_AMP_ICD9)),
    ("Major amputation, ICD-10-PCS procedure (prefix)", ", ".join(cfg.MAJOR_AMP_ICD10_PREFIXES)),
    ("Beta-blocker, vasodilating (Table S7)", ", ".join(cfg.VASODILATING_BB_DRUGS)),
    ("Beta-blocker, non-vasodilating (Table S7)", ", ".join(cfg.NONVASODILATING_BB_DRUGS)),
    ("Additional codes for any amputation, ICD-9-CM procedure", ", ".join(f"{c[:2]}.{c[2:]}" for c in cfg.OTHER_AMP_ICD9)),
    ("Additional codes for any amputation, ICD-10-PCS procedure (prefix)", ", ".join(cfg.OTHER_AMP_ICD10_PREFIXES)),
    ("Limb ischaemia or gangrene, ICD-9-CM diagnosis", ", ".join(f"{c[:3]}.{c[3:]}" for c in cfg.CLTI_ICD9)),
    ("Limb ischaemia or gangrene, ICD-10-CM diagnosis",
     "I70.22, I70.23, I70.24, I70.25, I70.26 (all subcodes), I96, E08.52, E09.52, E10.52, E11.52, E13.52"),
    ("Atherosclerotic PAD, ICD-9-CM / ICD-10-CM diagnosis", "440.2x / I70.2x"),
    # Covariates (added 2026-11-13, review v3 R3-M5): Quan et al. (2005) Charlson mapping, code prefixes
    ("Covariate: myocardial infarction, ICD-9-CM / ICD-10-CM (prefix)", "410, 412 / I21, I22, I25.2"),
    ("Covariate: congestive heart failure, ICD-9-CM / ICD-10-CM (prefix)",
     "428 / I09.9, I11.0, I13.0, I13.2, I25.5, I42.0, I42.5–I42.9, I43, I50, P29.0"),
    ("Covariate: cerebrovascular disease, ICD-9-CM / ICD-10-CM (prefix)",
     "430–438 / G45, G46, H34.0, I60–I69"),
    ("Covariate: chronic pulmonary disease, ICD-9-CM / ICD-10-CM (prefix)",
     "416.8, 416.9, 490–505 / I27.8, I27.9, J40–J47, J60–J67, J68.4, J70.1, J70.3"),
    ("Covariate: renal disease, ICD-9-CM / ICD-10-CM (prefix)",
     "582, 583, 585, 586, V42.0, V45.1 / I12.0, I13.1, N03.2–N03.7, N05.2–N05.7, N18, N19, N25.0, Z49.0–Z49.2, "
     "Z94.0, Z99.2"),
    ("Covariate: sepsis, ICD-9-CM / ICD-10-CM (prefix)",
     "038.0, 038.1x, 038.2, 038.3, 038.4x, 038.8, 038.9, 995.9 / A40, A41, R65.20, R65.21"),
    ("Covariate: Charlson Comorbidity Index",
     "Weighted sum of the 17 Charlson categories with the ICD-9-CM and ICD-10 coding algorithms of Quan et al. "
     "(2005), from all diagnosis codes of the admission; diabetes counted once at the higher weight"),
]
rp = pd.read_csv(os.path.join(RES, "review3_procedure_codes.csv"))  # script 16
for _, r in rp.iterrows():
    s8.append((f"{r.group} (Table S15), ICD-9-CM procedure", ", ".join(f"{c[:2]}.{c[2:]}" for c in r.icd9.split(", "))))
    s8.append((f"{r.group} (Table S15), ICD-10-PCS procedure",
               f"{int(r.icd10_pcs_n)} codes selected by title from d_icd_procedures, prefixes {r.icd10_pcs_prefixes}"))
pd.DataFrame(s8, columns=["Category", "Drug names or codes"]).to_csv(os.path.join(OUT, "TableS8_code_lists.csv"), index=False)

# --- Table S9: additional sensitivity analyses (script 12, review v2) -------
SCEN = {"A1b_no_contraindication": "No baseline contraindication",
        "A1c_any_class_48h": "Any studied class within 48 h",
        "A2b_index_stay_only": "Index stay only",
        "B4_first_admission": "First admission per patient",
        "B5_baseline_only_covariates": "Baseline-only covariates",
        "B6_calendar_era": "Calendar era",
        "B7_extended_censoring_model": "Extended censoring model"}
SHORT_OUT = {"afs": "AFS (primary)", "mortality": "Mortality", "amputation": "Major amputation"}
sr = pd.read_csv(os.path.join(RES, "sensitivity_review2.csv"))
s9 = []
for sc_name, lab in SCEN.items():
    for o in OUTCOMES:
        for c in COMPS:
            q = sr[(sr.scenario == sc_name) & (sr.outcome == o) & (sr.comparison == c)]
            if q.empty:
                continue
            r = q.iloc[0]
            s9.append({"Analysis": lab, "Outcome": SHORT_OUT[o], "Comparator": COMP_LABEL[c],
                       "Total n": f"{int(r.n_admissions)}",
                       "ACEI/ARB, events/n": f"{int(r.events_treated)}/{int(r.n_treated)}",
                       "Comparator, events/n": f"{int(r.events_control)}/{int(r.n_control)}",
                       "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper), "p": fmt_p(r.cox_p_value),
                       # 1-year RD not meaningful when follow-up ends at index discharge
                       "1-year RD (95% CI)": "–" if sc_name == "A2b_index_stay_only"
                       else fmt_rd(r.rd_1y, r.rd_1y_CI_lower, r.rd_1y_CI_upper),
                       "SMD ≥0.10": f"{int(r.n_covariates_imbalanced)}/{int(r.n_covariates)}"})
for c in COMPS:  # any (major or minor) amputation, script 15 task E3
    r = bbx[(bbx.task == "E3_anyamp") & (bbx.subset == "overall") & (bbx.comparison == c)].iloc[0]
    s9.append({"Analysis": "Any amputation", "Outcome": "Major or minor amputation", "Comparator": COMP_LABEL[c],
               "Total n": f"{int(r.n_admissions)}",
               "ACEI/ARB, events/n": f"{int(r.events_treated)}/{int(r.n_treated)}",
               "Comparator, events/n": f"{int(r.events_control)}/{int(r.n_control)}",
               "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper), "p": fmt_p(r.cox_p_value),
               "1-year RD (95% CI)": fmt_rd(r.rd_1y, r.rd_1y_CI_lower, r.rd_1y_CI_upper),
               "SMD ≥0.10": f"{int(r.n_covariates_imbalanced)}/{int(r.n_covariates)}"})
pd.DataFrame(s9).to_csv(os.path.join(OUT, "TableS9_additional_sensitivity.csv"), index=False)

# --- Table S10: strategy following, censoring probabilities, later admission (A1a, A2a) ---
a1 = pd.read_csv(os.path.join(RES, "sens_review2_A1a_strategy_following.csv"))
a2 = pd.read_csv(os.path.join(RES, "sens_review2_A2a_later_admission.csv"))
s10 = []
for c in COMPS:
    for o in OUTCOMES:
        r = a1[(a1.comparison == c) & (a1.outcome == o)].iloc[0]
        s10.append({"Comparator": COMP_LABEL[c], "Outcome": OUTCOME_LABEL[o], "Total n": f"{int(r.n_admissions)}",
                    "Event within grace period": f"{int(r.n_event_within_grace)}",
                    "Following ACEI/ARB": f"{int(r.n_follow_aceiarb)}",
                    "Following comparator": f"{int(r.n_follow_comparator)}",
                    "Following neither, n (%)": f"{int(r.n_follow_neither)} ({r.pct_neither:.1f})",
                    "P(uncensored), ACEI/ARB arm: min; median": f"{r.p_uncens_ACEI_ARB_min:.3f}; {r.p_uncens_ACEI_ARB_median:.2f}",
                    "P(uncensored), comparator arm: min; median": f"{r[f'p_uncens_{c}_min']:.3f}; {r[f'p_uncens_{c}_median']:.2f}"})
pd.DataFrame(s10).to_csv(os.path.join(OUT, "TableS10_strategy_following.csv"), index=False)
s10b = []
for c in COMPS:
    for _, r in a2[a2.comparison == c].iterrows():
        s10b.append({"Comparator": COMP_LABEL[c], "Arm": r.arm if r.arm == "ACEI/ARB" else COMP_LABEL[c],
                     "Alive at discharge, n": f"{int(r.n_alive_at_discharge)}",
                     "Later admission within 365 d, unweighted %": f"{r.pct_later_admission_365d_unweighted:.1f}",
                     "Later admission within 365 d, weighted %": f"{r.pct_later_admission_365d_weighted:.1f}"})
pd.DataFrame(s10b).to_csv(os.path.join(OUT, "TableS11_later_admission.csv"), index=False)

# --- Table S12: reasons for end of follow-up by arm (B9) --------------------
b9 = pd.read_csv(os.path.join(RES, "sens_review2_B9_end_of_followup.csv"))
s12 = []
for o in OUTCOMES:
    for c in COMPS:
        for _, r in b9[(b9.outcome == o) & (b9.comparison == c)].iterrows():
            s12.append({"Outcome": OUTCOME_LABEL[o], "Comparator": COMP_LABEL[c],
                        "Arm": r.arm if r.arm == "ACEI/ARB" else COMP_LABEL[c], "Clones": f"{int(r.clones)}",
                        "Artificially censored at 48 h": f"{int(r.artificially_censored_48h)}",
                        "Outcome event": f"{int(r.outcome_event)}",
                        "Death (censoring)": f"{int(r.death_censoring)}" if o == "amputation" else "–",
                        "End of follow-up (1 year after discharge)": f"{int(r.administrative_end)}"})
pd.DataFrame(s12).to_csv(os.path.join(OUT, "TableS12_end_of_followup.csv"), index=False)

# --- Table S13: subgroup analyses by limb ischaemia (script 15, tasks E1, E1b, E3) ---
MOD = {"clti": "Ischaemia/ gangrene", "athero_pad": "PAD"}
s13 = []
spec = [("clti", o, c, t) for o in OUTCOMES for c, t in [("CCB", "E1_CCB"), ("betablocker", "E1_betablocker"),
                                                         ("diuretic", "E1_diuretic")]]
spec += [("clti", "any_amputation", "betablocker", "E3_anyamp")]
spec += [("athero_pad", o, "betablocker", "E1b_athero") for o in OUTCOMES]
for mod, o, c, t in spec:
    q = bbx[(bbx.task == t) & (bbx.comparison == c) & (bbx.outcome == o)]
    yes, no = q[q.subset == f"{mod}=1"].iloc[0], q[q.subset == f"{mod}=0"].iloc[0]
    rat = q[q.subset.str.startswith("ratio")].iloc[0]
    ev = lambda r: f"{int(r.events_treated)}/{int(r.n_treated)}; {int(r.events_control)}/{int(r.n_control)}"
    s13.append({"Group": MOD[mod],
                "Outcome": "Any amputation" if o == "any_amputation" else SHORT_OUT[o],
                "Comparator": COMP_LABEL[c],
                "Present: events/n": ev(yes), "Present: HR (95% CI)": fmt_hr(yes.HR, yes.HR_CI_lower, yes.HR_CI_upper),
                "Absent: events/n": ev(no), "Absent: HR (95% CI)": fmt_hr(no.HR, no.HR_CI_lower, no.HR_CI_upper),
                "Ratio of HRs (95% CI)": fmt_hr(rat.ratio_of_HR, rat.ratio_CI_lower, rat.ratio_CI_upper),
                "p (int.)": fmt_p(rat.p_interaction)})
pd.DataFrame(s13).to_csv(os.path.join(OUT, "TableS13_subgroups_ischaemia.csv"), index=False)

# --- Tables S15-S18: review v3 analyses (scripts 16-17, 2026-11-13) -----------------------------
# S14 (timing of analyses) is maintained by hand in tables/TableS14_timing.csv.
SCEN3 = {"C1_admission_physiology": "Admission potassium and ED blood pressure",
         "C2a_exclude_early_limb_proc": "Early limb procedure excluded",
         "C2b_early_proc_in_ipcw": "Early limb procedures in censoring model",
         "C3_admin_censoring_365d": "Censoring at day 365",
         "C4_symmetric_comparator": "Symmetric strategies",
         "C5_bootstrap_reestimated": "Weights re-estimated in bootstrap",
         "C6_ipcw_only": "Censoring weights only"}
s3r = pd.read_csv(os.path.join(RES, "sensitivity_review3.csv"))
s15 = []
for sc_name, lab in SCEN3.items():
    for o in OUTCOMES:
        for c in COMPS:
            r = s3r[(s3r.scenario == sc_name) & (s3r.outcome == o) & (s3r.comparison == c)].iloc[0]
            s15.append({"Analysis": lab, "Outcome": SHORT_OUT[o], "Comparator": COMP_LABEL[c],
                        "Total n": f"{int(r.n_admissions)}",
                        "ACEI/ARB, events/n": f"{int(r.events_treated)}/{int(r.n_treated)}",
                        "Comparator, events/n": f"{int(r.events_control)}/{int(r.n_control)}",
                        "HR (95% CI)": fmt_hr(r.HR, r.HR_CI_lower, r.HR_CI_upper), "p": fmt_p(r.cox_p_value),
                        "1-year RD (95% CI)": fmt_rd(r.rd_1y, r.rd_1y_CI_lower, r.rd_1y_CI_upper),
                        "SMD ≥0.10": f"{int(r.n_covariates_imbalanced)}/{int(r.n_covariates)}"})
# Multiple imputation of outpatient blood pressure (script 18, 2026-11-13): Rubin's rules with the robust SE;
# events and n are those of the primary analysis (imputation changes the weights only, not the clones).
mi = pd.read_csv(os.path.join(RES, "sensitivity_mi_bp.csv"))
pr = pd.read_csv(os.path.join(RES, "primary_results_ALL.csv"))
for o in OUTCOMES:
    for c in COMPS:
        r = mi[(mi.outcome == o) & (mi.comparison == c)].iloc[0]
        p0 = pr[(pr.outcome == o) & (pr.comparison == f"ACEI_ARB_vs_{c}")].iloc[0]
        s15.append({"Analysis": "Multiple imputation of outpatient blood pressure", "Outcome": SHORT_OUT[o],
                    "Comparator": COMP_LABEL[c], "Total n": "3753" if o == "mortality" else "3712",
                    "ACEI/ARB, events/n": f"{int(p0.events_treated)}/{int(p0.n_treated)}",
                    "Comparator, events/n": f"{int(p0.events_control)}/{int(p0.n_control)}",
                    "HR (95% CI)": fmt_hr(r.HR_mi, r.HR_mi_CI_lower, r.HR_mi_CI_upper), "p": fmt_p(r.p_mi),
                    "1-year RD (95% CI)": "–",
                    "SMD ≥0.10": f"{int(r.max_n_imbalanced)}/{int(r.n_cov)}"})
pd.DataFrame(s15).to_csv(os.path.join(OUT, "TableS15_further_sensitivity.csv"), index=False)

# S16: treatment patterns, pre-admission therapy and early limb procedures by strategy
d1 = pd.read_csv(os.path.join(RES, "review3_D1_persistence.csv"))
d3 = pd.read_csv(os.path.join(RES, "review3_D3_ed_by_arm.csv"))
d4 = pd.read_csv(os.path.join(RES, "review3_D4_early_procedures_by_arm.csv"))
CLS_LAB = {"CCB": "ccb", "betablocker": "betablocker", "diuretic": "diuretic"}
s16 = []
for c in COMPS:
    for arm in ["ACEI/ARB", c]:
        if arm == "ACEI/ARB" and c != "CCB":
            continue  # the ACEI/ARB strategy is identical in the three comparisons
        a1 = d1[(d1.comparison == c) & (d1.arm == arm)].iloc[0]
        a3 = d3[(d3.comparison == c) & (d3.arm == arm)].iloc[0]
        a4 = d4[(d4.comparison == c) & (d4.arm == arm)].iloc[0]
        own = "aceiarb" if arm == "ACEI/ARB" else CLS_LAB[c]
        s16.append({"Strategy": "ACEI/ARB" if arm == "ACEI/ARB" else COMP_LABEL[c],
                    "n at 48 h": f"{int(a3.n_following)}",
                    "ED medication list, n (%)": f"{int(a3.n_medrecon)} ({a3.pct_medrecon:.1f})",
                    "ACEI/ARB before admission, %": f"{a3.pct_pre_aceiarb:.1f}",
                    "Own class before admission, %": f"{a3[f'pct_pre_{own}']:.1f}",
                    "ED triage SBP, mmHg (available %)": f"{a3.ed_sbp_mean:.0f} ({a3.pct_ed_sbp_available:.1f})",
                    "Own class in last 24 h, % (weighted)": f"{a1.pct_anchor_last24h:.1f} ({a1.pct_anchor_last24h_weighted:.1f})",
                    "ACEI/ARB after 48 h, % (weighted)": "–" if arm == "ACEI/ARB" else
                    f"{a1.pct_aceiarb_after48h:.1f} ({a1.pct_aceiarb_after48h_weighted:.1f})",
                    "Early limb procedure, % (weighted)": f"{a4.pct_early_limb_proc:.1f} ({a4.pct_early_limb_proc_weighted:.1f})",
                    "Revascularization in index stay, % (weighted)": f"{a4.pct_index_revasc:.1f} ({a4.pct_index_revasc_weighted:.1f})"})
pd.DataFrame(s16).to_csv(os.path.join(OUT, "TableS16_treatment_patterns.csv"), index=False)

# S17: baseline characteristics of the full cohort and of admissions with no studied class within 48 h
t2x = pd.read_csv(os.path.join(RES, "review3_D2_table2_extra_columns.csv")).set_index("variable")
lab17 = {"Admissions, n": "Admissions, n", "Patients, n": "Patients, n"}
for v in t2x.index:
    if v not in lab17:
        lab17[v] = COV_LABEL.get(v, v) + (", mean (SD)" if v in CONTINUOUS else ", %")
t2x.index = [lab17[v] for v in t2x.index]
t2x.index.name = "Characteristic"
t2x.to_csv(os.path.join(OUT, "TableS17_full_cohort.csv"))

# S18: effective sample size, P(uncensored) by contraindication, amputation-ascertainment bias analysis
ess = pd.read_csv(os.path.join(RES, "review3_D2_ess.csv"))
pu = pd.read_csv(os.path.join(RES, "review3_D2_p_uncensored_by_contraindication.csv"))
qba = pd.read_csv(os.path.join(RES, "review3_D8_qba_amputation.csv"))
s18 = []
for c in COMPS:
    for arm in ["ACEI/ARB", c]:
        e = ess[(ess.comparison == c) & (ess.arm == arm) & (ess.outcome == "afs")].iloc[0]
        p0 = pu[(pu.comparison == c) & (pu.arm == arm) & (pu.baseline_contraindication == 0)].iloc[0]
        p1 = pu[(pu.comparison == c) & (pu.arm == arm) & (pu.baseline_contraindication == 1)].iloc[0]
        q = qba[(qba.comparison == c) & (qba.arm == arm)].iloc[0]
        s18.append({"Comparator": COMP_LABEL[c], "Arm": "ACEI/ARB" if arm == "ACEI/ARB" else COMP_LABEL[c],
                    "Retained clones, n": f"{int(e.n_retained)}", "Effective sample size": f"{e.kish_ess:.0f}",
                    "P(uncensored) without contra- indication: min; median": f"{p0.p_min:.3f}; {p0.p_median:.2f}",
                    "P(uncensored) with contra- indication: min; median": f"{p1.p_min:.3f}; {p1.p_median:.2f}",
                    "Major amputation within 1 year, observed %": f"{100 * q.risk_observed:.1f}",
                    "Later admission, %": f"{100 * q.p_later_admission:.1f}",
                    "Major amputation within 1 year, corrected %": f"{100 * q.risk_corrected:.1f}"})
pd.DataFrame(s18).to_csv(os.path.join(OUT, "TableS18_positivity_ascertainment.csv"), index=False)

for f in sorted(os.listdir(OUT)):
    if f.endswith(".csv"):
        print(f, len(pd.read_csv(os.path.join(OUT, f))), "rows")
