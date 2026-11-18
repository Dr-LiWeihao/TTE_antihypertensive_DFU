"""
04_outcomes.py
---------------
REBUILT 2026-08-31 (REVISION 3 + composite-outcome addendum).

Three outcome families, all time-to-event from T0 = admittime (CLAUDE.md
"Outcome hierarchy REVISION" 2026-08-31 -- AFS is now PRIMARY, mortality
and amputation-alone are SECONDARY):
  1. afs_event / afs_days -- amputation-free survival (PRIMARY): the
     earlier of death (any cause) or a MAJOR amputation occurring
     STRICTLY AFTER T0. Admissions with a major amputation AT T0 (same
     calendar day as admission -- procedures_icd chartdate is date-only,
     so day-0 events can't be ordered relative to the actual admission
     moment) are EXCLUDED from the AFS analysis (afs_eligible = False).
  2. mortality_event / mortality_days -- all-cause mortality (SECONDARY):
     same in-hospital + 1-year-post-discharge definition as project_009,
     simply days-from-admission (T0 = admittime, no re-anchoring offset).
  3. amputation_only_event / amputation_only_days -- major amputation
     ALONE (SECONDARY, new 2026-08-31): a post-T0 major amputation,
     follow-up CENSORED AT DEATH (or administrative censoring if neither
     occurs) -- i.e. death is treated as a censoring event for this
     outcome, NOT modeled as a formal competing risk (Fine-Gray
     subdistribution hazard). This is a real simplification, consistent
     with project_009's own amputation-outcome convention
     (scripts/03_amputation_phenotype.py), and is stated as a limitation:
     since death precludes amputation, treating it as ordinary censoring
     is informative censoring, not the fully rigorous approach. Uses the
     SAME afs_eligible flag (excludes day-0 amputations) as AFS, since
     the exclusion rationale is identical.

Output: data/outcomes.parquet
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
tx = eligible.copy()

# eligible_cohort.parquet already carries a `dod` column inherited from
# project_009's cohort file -- drop before re-merging fresh from
# hosp/patients to avoid a dod_x/dod_y suffix collision (same bug
# pattern documented in the pre-rebuild version of this script).
if "dod" in tx.columns:
    tx = tx.drop(columns=["dod"])
pts = pd.read_csv(mimic("hosp", "patients"), usecols=["subject_id", "dod"], parse_dates=["dod"])
tx = tx.merge(pts, on="subject_id", how="left")

# dod-before-admission cleaning (unreliable death-registry-linkage errors,
# same finding as project_009)
dod_before_admission = tx["dod"].notna() & (tx["dod"] < tx["admittime"])
n_unreliable = dod_before_admission.sum()
log(f"Unreliable dod (predates admission): {n_unreliable}")
tx["dod_clean"] = tx["dod"].where(~dod_before_admission)

# same-day dod-vs-discharge granularity fix (dod is date-only)
days_disch_to_dod = (tx["dod_clean"] - tx["dischtime"]).dt.total_seconds() / 86400
days_disch_to_dod_clipped = days_disch_to_dod.clip(lower=0)

died_within_1y_postdisch = (tx["hospital_expire_flag"] == 0) & tx["dod_clean"].notna() & (days_disch_to_dod_clipped <= 365)
tx["mortality_event"] = (tx["hospital_expire_flag"] == 1) | died_within_1y_postdisch

los_days_calc = (tx["dischtime"] - tx["admittime"]).dt.total_seconds() / 86400
days_to_dod_clipped = los_days_calc + days_disch_to_dod_clipped

# REVISION 3: T0 = admittime, so days-from-admit IS days-from-T0 directly
# (no 48h re-anchoring offset needed).
mortality_days = np.where(
    tx["hospital_expire_flag"] == 1,
    (tx["deathtime"] - tx["admittime"]).dt.total_seconds() / 86400,
    np.where(died_within_1y_postdisch, days_to_dod_clipped, los_days_calc + 365),
)
tx["mortality_days_from_t0"] = mortality_days

log(f"mortality_event: {tx.mortality_event.sum():,} ({100*tx.mortality_event.mean():.1f}%)")
log(f"mortality_days_from_t0 range: {tx.mortality_days_from_t0.min():.2f} to {tx.mortality_days_from_t0.max():.2f}")
assert (tx.mortality_days_from_t0 > 0).all(), "Non-positive time-from-T0 found in mortality outcome"

# --- Amputation-free survival ---
# REVISED 2026-11-10 (pre-submission review, notes/2026-11-10_druglist_amputation_rerun.md).
# Before: major amputations were taken only from the INDEX admission (hadm_id match),
# while AFS / amputation follow-up ran to discharge + 365 d, so post-discharge person-time
# was counted as amputation-free without ascertainment. Now: major amputations are taken
# from ALL of the patient's MIMIC-IV admissions (subject_id match, procedures_icd
# chartdate), from the admission calendar day up to the end of the mortality follow-up
# window (death or discharge + 365 d). Code list completed in config.py
# (MAJOR_AMP_ICD9 / MAJOR_AMP_ICD10_PREFIXES). Day-0 rule unchanged: a major amputation
# on the admission calendar day cannot be ordered relative to admission (chartdate is
# date-only) -> admission excluded from the amputation analyses (afs_eligible=False).
proc = pd.read_csv(
    mimic("hosp", "procedures_icd"),
    usecols=["subject_id", "hadm_id", "chartdate", "icd_code", "icd_version"],
    parse_dates=["chartdate"],
    dtype={"subject_id": "int64", "hadm_id": "int64", "icd_version": "int8", "icd_code": str},
)
proc["icd_code"] = proc["icd_code"].str.strip()
proc = proc[proc.subject_id.isin(set(tx.subject_id))].reset_index(drop=True)
is_major = ((proc.icd_version == 9) & proc.icd_code.isin(MAJOR_AMP_ICD9)) | \
           ((proc.icd_version == 10) & proc.icd_code.str.startswith(MAJOR_AMP_ICD10_PREFIXES))
amp_all = proc.loc[is_major, ["subject_id", "hadm_id", "chartdate"]].rename(columns={"hadm_id": "amp_hadm_id"})

cand = tx[["hadm_id", "subject_id", "admittime", "dischtime", "mortality_days_from_t0"]].merge(
    amp_all, on="subject_id", how="inner")
cand["d"] = (cand["chartdate"] - cand["admittime"]).dt.total_seconds() / 86400
# Keep: (a) amputations from the admission calendar day onward, or (b) any amputation
# recorded in the INDEX admission itself even if its chartdate precedes the admission day
# (pre-admission ED/observation time; previously clipped to day 0 and excluded -- kept
# that way so the day-0 exclusion is unchanged). All must fall within follow-up.
cand = cand[((cand["chartdate"] >= cand["admittime"].dt.normalize()) | (cand["amp_hadm_id"] == cand["hadm_id"])) &
            (cand["d"] <= cand["mortality_days_from_t0"])]
first = cand.sort_values("d").groupby("hadm_id").first()
tx = tx.merge(first[["chartdate", "amp_hadm_id"]].rename(columns={"chartdate": "major_amp_date"}),
              left_on="hadm_id", right_index=True, how="left")
tx["days_to_major_amp"] = (tx["major_amp_date"] - tx["admittime"]).dt.total_seconds() / 86400
n_neg = (tx["days_to_major_amp"] < 0).sum()
tx["days_to_major_amp"] = tx["days_to_major_amp"].clip(lower=0)
same_adm = tx["amp_hadm_id"] == tx["hadm_id"]
log(f"\nFirst major amputation within follow-up: {tx['major_amp_date'].notna().sum():,} "
    f"({100*tx['major_amp_date'].notna().mean():.1f}%), same-day-clipped: {n_neg:,}")
log(f"  recorded in the index admission: {same_adm.sum():,}; "
    f"in a later admission within follow-up: {(tx['major_amp_date'].notna() & ~same_adm).sum():,}")

# Pre-T0 (day-0) major amputation -> excluded from AFS analysis
tx["amp_at_t0"] = tx["major_amp_date"].notna() & (tx["days_to_major_amp"] <= 0)
n_amp_at_t0 = tx["amp_at_t0"].sum()
log(f"Major amputations at T0 (same calendar day as admission, EXCLUDED "
    f"from AFS analysis per CLAUDE.md composite-outcome addendum): {n_amp_at_t0:,}")

tx["afs_eligible"] = ~tx["amp_at_t0"]

# AFS event = death OR post-T0 major amputation, whichever first
has_post_t0_amp = tx["major_amp_date"].notna() & (tx["days_to_major_amp"] > 0)
tx["afs_event"] = tx["mortality_event"] | has_post_t0_amp
afs_days_candidates = pd.concat([
    tx["mortality_days_from_t0"].where(tx["mortality_event"]),
    tx["days_to_major_amp"].where(has_post_t0_amp),
], axis=1)
tx["afs_days_from_t0"] = afs_days_candidates.min(axis=1, skipna=True)
# Admissions with neither event: AFS-censored at the same time as the
# mortality censoring time (administrative censoring, discharge+1y window)
tx["afs_days_from_t0"] = tx["afs_days_from_t0"].fillna(tx["mortality_days_from_t0"])

log(f"\nAFS analysis population (PRIMARY): {tx['afs_eligible'].sum():,} / {len(tx):,} "
    f"(excludes {n_amp_at_t0:,} day-0 major amputations)")
afs_pop = tx[tx["afs_eligible"]]
log(f"afs_event (among AFS-eligible): {afs_pop.afs_event.sum():,} "
    f"({100*afs_pop.afs_event.mean():.1f}%)")
log(f"  of which from amputation (not death): "
    f"{(afs_pop.afs_event & ~afs_pop.mortality_event).sum():,}")
assert (afs_pop.afs_days_from_t0 > 0).all(), "Non-positive AFS time-from-T0 found"

# --- Amputation-alone (SECONDARY, new 2026-08-31) ---
tx["amputation_only_event"] = has_post_t0_amp
tx["amputation_only_days_from_t0"] = tx["days_to_major_amp"].where(has_post_t0_amp, tx["mortality_days_from_t0"])
amp_pop = tx[tx["afs_eligible"]]
log(f"\namputation_only_event (SECONDARY, among AFS-eligible pop, death "
    f"treated as censoring -- see docstring limitation): "
    f"{amp_pop.amputation_only_event.sum():,} ({100*amp_pop.amputation_only_event.mean():.1f}%)")
assert (amp_pop.amputation_only_days_from_t0 > 0).all(), "Non-positive amputation-only time-from-T0 found"

out = tx.drop(columns=["dod", "major_amp_date"])
out.to_parquet(data("outcomes.parquet"), index=False)
log("\nSaved: data/outcomes.parquet")

with open(log_path("04_outcomes.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
