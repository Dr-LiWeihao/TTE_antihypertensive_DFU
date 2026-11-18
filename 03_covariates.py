"""
03_covariates.py
------------------
REBUILT 2026-08-31 (REVISION 3 + Assignment Procedure Revision, both
folded into this same rebuild per CLAUDE.md).

Baseline confounders for the propensity model:
  - CKD/nephropathy stage, heart failure, MI, cerebrovascular, chronic
    pulmonary, sepsis, CCI (reuse project_009's Quan(2005) mapping)
  - Race, age, gender
  - Baseline (admission) creatinine
  - NEW: baseline blood pressure (omr, most recent within
    BP_LOOKBACK_DAYS pre-admission) -- median-imputed with a "bp_unknown"
    indicator (locked decision, CLAUDE.md "Assignment procedure REVISION
    2026-08-28")
  - NEW: illness-severity proxies -- collapsed admission_type
    (emergency/urgent vs elective/scheduled) + ICU-stay-during-admission
    flag (both 100% coverage; chartevents-based severity proxies are
    ICU-only, ruled infeasible hospital-wide, see CLAUDE.md)

T0 = admittime (REVISION 3). Baseline creatinine window changed from "at
or before the old 48h T0" (98.8% coverage) to "first creatinine value
charted during the admission" (admission-presenting value) -- moving T0
back to admittime makes "before T0" nearly empty, so this is redefined as
the standard EHR-study convention of "first available inpatient value"
rather than an arbitrary new cutoff. Logged explicitly below.

Output: data/covariates.parquet
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import *

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

eligible = pd.read_parquet(data("eligible_cohort.parquet"))
hadm_set = set(eligible.hadm_id)
log(f"Eligible admissions: {len(eligible):,}")

# --- Comorbidities / CCI: reuse project_009's Quan(2005) computation ---
cci = pd.read_parquet(data("comorbidity_cci.parquet"))
cci = cci[cci.hadm_id.isin(hadm_set)]
log(f"CCI/comorbidity rows matched: {len(cci):,} / {len(eligible):,}")

# --- Race, gender, admission_type, age already in eligible_cohort (from
#     project_009's cohort file) ---
def simplify_race(r):
    if pd.isna(r):
        return "Unknown"
    r = r.upper()
    if "WHITE" in r:
        return "White"
    if "BLACK" in r:
        return "Black"
    if "ASIAN" in r:
        return "Asian"
    if "HISPANIC" in r or "LATINO" in r:
        return "Hispanic"
    return "Other/Unknown"

eligible["race_simplified"] = eligible["race"].apply(simplify_race)

# --- NEW: illness severity -- collapsed admission_type ---
EMERGENT_TYPES = {"EW EMER.", "EU OBSERVATION", "OBSERVATION ADMIT", "URGENT",
                   "DIRECT OBSERVATION", "DIRECT EMER.", "AMBULATORY OBSERVATION"}
eligible["emergency_admission"] = eligible["admission_type"].isin(EMERGENT_TYPES).astype(int)
log(f"\nEmergency/urgent admission_type: {eligible.emergency_admission.sum():,} "
    f"({100*eligible.emergency_admission.mean():.1f}%)")

# --- NEW: illness severity -- ICU stay during this admission ---
icu = pd.read_csv(mimic("icu", "icustays"), usecols=["hadm_id"], dtype={"hadm_id": "int32"})
icu_hadm = set(icu.hadm_id)
eligible["icu_stay_flag"] = eligible["hadm_id"].isin(icu_hadm).astype(int)
log(f"ICU stay during admission: {eligible.icu_stay_flag.sum():,} "
    f"({100*eligible.icu_stay_flag.mean():.1f}%)")

# --- NEW: baseline blood pressure (omr, pre-admission window) ---
omr = pd.read_csv(
    mimic("hosp", "omr"),
    usecols=["subject_id", "chartdate", "result_name", "result_value"],
    dtype={"subject_id": "int32", "result_name": str, "result_value": str},
    parse_dates=["chartdate"],
)
omr_bp = omr[(omr.result_name == "Blood Pressure") & omr.subject_id.isin(set(eligible.subject_id))].copy()
bp_split = omr_bp["result_value"].str.split("/", expand=True)
omr_bp["sbp"] = pd.to_numeric(bp_split[0], errors="coerce")
omr_bp["dbp"] = pd.to_numeric(bp_split[1], errors="coerce")
omr_bp = omr_bp.dropna(subset=["sbp", "dbp"])

adm_lookup = eligible[["hadm_id", "subject_id", "admittime"]].copy()
bp_merged = adm_lookup.merge(omr_bp[["subject_id", "chartdate", "sbp", "dbp"]], on="subject_id", how="left")
bp_merged["days_before_admit"] = (bp_merged["admittime"] - bp_merged["chartdate"]).dt.total_seconds() / 86400
bp_window = bp_merged[(bp_merged["days_before_admit"] >= 0) & (bp_merged["days_before_admit"] <= BP_LOOKBACK_DAYS)]
bp_most_recent = bp_window.sort_values("days_before_admit").groupby("hadm_id").first()[["sbp", "dbp"]]
bp_most_recent.columns = ["baseline_sbp", "baseline_dbp"]

n_bp_available = len(bp_most_recent)
log(f"\nBaseline BP available (omr, within {BP_LOOKBACK_DAYS}d pre-admission): "
    f"{n_bp_available:,} / {len(eligible):,} ({100*n_bp_available/len(eligible):.1f}%)")

# --- Baseline (admission) creatinine: first value charted this admission ---
crea_itemids = [50912, 52546]
chunks = pd.read_csv(
    mimic("hosp", "labevents"),
    usecols=["hadm_id", "itemid", "charttime", "valuenum"],
    dtype={"hadm_id": "float64", "itemid": "int32"},  # float64, NOT float32: see 02 (bug fixed 2026-10-01)
    parse_dates=["charttime"],
    chunksize=3_000_000,
)
crea_rows = []
for ch in chunks:
    ch = ch[ch.hadm_id.notna() & ch.itemid.isin(crea_itemids) & ch.valuenum.notna()]
    if ch.empty:
        continue
    ch["hadm_id"] = ch["hadm_id"].astype("int32")
    ch = ch[ch.hadm_id.isin(hadm_set)]
    if not ch.empty:
        crea_rows.append(ch)
crea_all = pd.concat(crea_rows, ignore_index=True)

# BUG GUARD: a small fraction of labevents rows linked to our hadm_ids
# carry a charttime far BEFORE admittime (seen: up to ~95 years earlier)
# -- a MIMIC hadm_id-linkage artifact, not a real pre-admission draw.
# Taking the naive first-by-charttime would let these corrupt the
# "baseline" value. Require the lab to be charted AT OR AFTER admission.
admittime_lookup_crea = eligible.set_index("hadm_id")["admittime"]
crea_all["admittime"] = crea_all["hadm_id"].map(admittime_lookup_crea)
crea_all["hours_from_admit"] = (crea_all["charttime"] - crea_all["admittime"]).dt.total_seconds() / 3600
n_bad_linkage = (crea_all["hours_from_admit"] < 0).sum()
log(f"Creatinine rows with charttime before admittime (bad hadm_id linkage, dropped): "
    f"{n_bad_linkage:,} / {len(crea_all):,}")
crea_all = crea_all[crea_all["hours_from_admit"] >= 0]

crea_first = crea_all.sort_values("charttime").groupby("hadm_id").first()["valuenum"].rename("baseline_creatinine")
log(f"Baseline (first-charted, post-admission) creatinine available: "
    f"{len(crea_first):,} / {len(eligible):,} ({100*len(crea_first)/len(eligible):.1f}%)")
log("NOTE (2026-10-01): the ~48% coverage reported between 2026-08-31 and "
    "2026-10-01 was an ARTIFACT of reading hadm_id as float32 (odd hadm_ids "
    "rounded to even and failed to match). With float64 coverage is ~97%. "
    "The impute + indicator handling below is kept for the remaining missing values.")

# --- Assemble ---
covariates = eligible[["hadm_id", "subject_id", "age_at_admission", "gender",
                        "race_simplified", "emergency_admission", "icu_stay_flag"]].copy()
covariates = covariates.merge(cci, on="hadm_id", how="left")
covariates = covariates.merge(bp_most_recent, on="hadm_id", how="left")
covariates = covariates.merge(crea_first, on="hadm_id", how="left")

# MISSINGNESS DECISION (locked, CLAUDE.md "Assignment procedure REVISION
# 2026-08-28"): median-impute baseline BP + explicit "bp_unknown"
# indicator, rather than complete-case restriction. Same principle now
# applied to baseline creatinine (~3% missing after the 2026-10-01
# float32 hadm_id fix; the earlier ~48% "coverage" was that bug).
covariates["bp_unknown"] = covariates["baseline_sbp"].isna().astype(int)
covariates["baseline_sbp"] = covariates["baseline_sbp"].fillna(covariates["baseline_sbp"].median())
covariates["baseline_dbp"] = covariates["baseline_dbp"].fillna(covariates["baseline_dbp"].median())
covariates["creatinine_unknown"] = covariates["baseline_creatinine"].isna().astype(int)
covariates["baseline_creatinine"] = covariates["baseline_creatinine"].fillna(covariates["baseline_creatinine"].median())

log(f"\nAssembled covariates: {len(covariates):,} rows")
for col in ["mi", "chf", "cerebrovascular", "chronic_pulmonary", "renal", "sepsis"]:
    log(f"{col}: {100*covariates[col].mean():.1f}%")
log(f"bp_unknown: {covariates.bp_unknown.sum():,} ({100*covariates.bp_unknown.mean():.1f}%)")
log(f"creatinine_unknown: {covariates.creatinine_unknown.sum():,} ({100*covariates.creatinine_unknown.mean():.1f}%)")
log(f"race distribution:\n{covariates.race_simplified.value_counts().to_string()}")

covariates.to_parquet(data("covariates.parquet"), index=False)
log("\nSaved: data/covariates.parquet")

with open(log_path("03_covariates.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
