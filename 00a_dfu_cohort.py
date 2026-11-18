"""
Diabetic foot cohort extraction from MIMIC-IV v3.1.

Case definition (documented, tiered — see notes/cohort_definition.md):
  Tier A - Diabetic foot ulcer
    ICD-10: E08.621, E09.621, E10.621, E11.621, E13.621  (diabetes + foot ulcer, single combined code)
    ICD-9 : (250.60-250.63 OR 250.70-250.73) AND (707.14 OR 707.15) in the SAME hadm_id
            [ICD-9 has no combined code; diabetic foot ulcer is conventionally proxied by
             co-occurrence of a diabetes-with-complication code and a foot ulcer code]
  Tier B - Diabetic foot gangrene
    ICD-10: E08.52, E09.52, E10.52, E11.52, E13.52  (diabetes + peripheral angiopathy WITH gangrene)
    ICD-9 : 250.70-250.73 AND 785.4 in the SAME hadm_id
            [785.4 "gangrene" is not foot-specific in ICD-9; paired with diabetic peripheral
             circulatory disorder this is the standard proxy for diabetic (foot) gangrene, but
             carries more site ambiguity than Tier A/ICD-10 codes]

  CORE COHORT = Tier A UNION Tier B (any admission with the diagnosis, any position)

  Reported separately, NOT in core cohort (documented, not silently dropped):
    Tier C - Diabetic peripheral angiopathy WITHOUT gangrene (ICD-10 E*.51) — vascular disease,
             not necessarily an active foot ulcer/gangrene event.
    Other skin ulcer (ICD-10 E*.622) — site not specified as foot per the code itself.
    Z86.31 "personal history of diabetic foot ulcer" — history code, not an active diagnosis.

Output:
  data/diabetic_foot_cohort_admissions.parquet  (admission-level, all tiers flagged)
  data/diabetic_foot_cohort_patients.parquet    (patient-level, first qualifying admission)
  logs/01_cohort_extraction.log
"""
import sys
import pandas as pd
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, DATA_DIR, LOG_DIR
BASE = BASE_DIR

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("Diabetic foot cohort extraction — MIMIC-IV v3.1")
log("=" * 70)

# ---------------------------------------------------------------
# 1. Load diagnosis dictionary and diagnoses_icd
# ---------------------------------------------------------------
d_icd = pd.read_csv(f"{BASE}/hosp/d_icd_diagnoses.csv.gz", dtype=str)

diag = pd.read_csv(
    f"{BASE}/hosp/diagnoses_icd.csv.gz",
    dtype={"subject_id": "int32", "hadm_id": "int32", "seq_num": "float32",
           "icd_code": str, "icd_version": "int8"},
)
log(f"diagnoses_icd loaded: {len(diag):,} rows, {diag['hadm_id'].nunique():,} admissions, "
    f"{diag['subject_id'].nunique():,} patients")

diag["icd_code"] = diag["icd_code"].str.strip()

# ---------------------------------------------------------------
# 2. Define code sets
# ---------------------------------------------------------------
icd10_tierA = ["E08621", "E09621", "E10621", "E11621", "E13621"]
icd10_tierB = ["E0852", "E0952", "E1052", "E1152", "E1352"]
icd10_tierC = ["E0851", "E0951", "E1051", "E1151", "E1351"]          # reported, not core
icd10_other_ulcer = ["E08622", "E09622", "E10622", "E11622", "E13622"]  # reported, not core
icd10_history = ["Z8631"]                                             # reported, not core

icd9_diab_neuro = ["25060", "25061", "25062", "25063"]                # neuro manifestations
icd9_diab_periph = ["25070", "25071", "25072", "25073"]               # peripheral circulatory
icd9_foot_ulcer = ["70714", "70715"]                                  # heel/midfoot, other part of foot
icd9_gangrene = ["7854"]

# sanity check: confirm codes exist in dictionary
all_direct = icd10_tierA + icd10_tierB + icd10_tierC + icd10_other_ulcer + icd10_history
missing = [c for c in all_direct if c not in set(d_icd["icd_code"])]
if missing:
    log(f"WARNING: codes not found in d_icd_diagnoses: {missing}")
else:
    log("All ICD-10 direct codes verified against d_icd_diagnoses.")

# ---------------------------------------------------------------
# 3. ICD-10 direct-code tiers
# ---------------------------------------------------------------
hadm_tierA_icd10 = set(diag.loc[(diag.icd_version == 10) & diag.icd_code.isin(icd10_tierA), "hadm_id"])
hadm_tierB_icd10 = set(diag.loc[(diag.icd_version == 10) & diag.icd_code.isin(icd10_tierB), "hadm_id"])
hadm_tierC_icd10 = set(diag.loc[(diag.icd_version == 10) & diag.icd_code.isin(icd10_tierC), "hadm_id"])
hadm_other_ulcer_icd10 = set(diag.loc[(diag.icd_version == 10) & diag.icd_code.isin(icd10_other_ulcer), "hadm_id"])
hadm_history_icd10 = set(diag.loc[(diag.icd_version == 10) & diag.icd_code.isin(icd10_history), "hadm_id"])

log(f"\nICD-10 Tier A (foot ulcer, direct code):      {len(hadm_tierA_icd10):,} admissions")
log(f"ICD-10 Tier B (gangrene, direct code):         {len(hadm_tierB_icd10):,} admissions")
log(f"ICD-10 Tier C (angiopathy, no gangrene, excl): {len(hadm_tierC_icd10):,} admissions")
log(f"ICD-10 other skin ulcer (site ambiguous, excl):{len(hadm_other_ulcer_icd10):,} admissions")
log(f"ICD-10 Z86.31 history of DFU (excl, not active):{len(hadm_history_icd10):,} admissions")

# ---------------------------------------------------------------
# 4. ICD-9 combination-code tiers (co-occurrence within same hadm_id)
# ---------------------------------------------------------------
icd9 = diag[diag.icd_version == 9]
hadm_diab_neuro9 = set(icd9.loc[icd9.icd_code.isin(icd9_diab_neuro), "hadm_id"])
hadm_diab_periph9 = set(icd9.loc[icd9.icd_code.isin(icd9_diab_periph), "hadm_id"])
hadm_foot_ulcer9 = set(icd9.loc[icd9.icd_code.isin(icd9_foot_ulcer), "hadm_id"])
hadm_gangrene9 = set(icd9.loc[icd9.icd_code.isin(icd9_gangrene), "hadm_id"])

hadm_diab_any9 = hadm_diab_neuro9 | hadm_diab_periph9
hadm_tierA_icd9 = hadm_diab_any9 & hadm_foot_ulcer9
hadm_tierB_icd9 = hadm_diab_periph9 & hadm_gangrene9

log(f"\nICD-9 Tier A (diabetes[250.6x/250.7x] + foot ulcer[707.14/15] co-occur): {len(hadm_tierA_icd9):,} admissions")
log(f"ICD-9 Tier B (diabetes[250.7x] + gangrene[785.4] co-occur):               {len(hadm_tierB_icd9):,} admissions")

# ---------------------------------------------------------------
# 5. Assemble core cohort
# ---------------------------------------------------------------
hadm_tierA = hadm_tierA_icd10 | hadm_tierA_icd9
hadm_tierB = hadm_tierB_icd10 | hadm_tierB_icd9
hadm_core = hadm_tierA | hadm_tierB

log(f"\nCore cohort (Tier A ∪ Tier B), admission-level: {len(hadm_core):,} admissions")

cohort_adm = pd.DataFrame({"hadm_id": sorted(hadm_core)})
cohort_adm["tier_A_ulcer"] = cohort_adm.hadm_id.isin(hadm_tierA)
cohort_adm["tier_B_gangrene"] = cohort_adm.hadm_id.isin(hadm_tierB)
cohort_adm["tier_C_angiopathy_only"] = cohort_adm.hadm_id.isin(hadm_tierC_icd10)
cohort_adm["flag_other_skin_ulcer"] = cohort_adm.hadm_id.isin(hadm_other_ulcer_icd10)
cohort_adm["flag_history_dfu_code"] = cohort_adm.hadm_id.isin(hadm_history_icd10)
cohort_adm["source_icd9"] = cohort_adm.hadm_id.isin(hadm_tierA_icd9 | hadm_tierB_icd9)
cohort_adm["source_icd10"] = cohort_adm.hadm_id.isin(hadm_tierA_icd10 | hadm_tierB_icd10)

# ---------------------------------------------------------------
# 6. Merge admissions + patients for demographics, LOS, mortality
# ---------------------------------------------------------------
adm = pd.read_csv(
    f"{BASE}/hosp/admissions.csv.gz",
    parse_dates=["admittime", "dischtime", "deathtime"],
    dtype={"subject_id": "int32", "hadm_id": "int32", "hospital_expire_flag": "int8"},
)
pts = pd.read_csv(
    f"{BASE}/hosp/patients.csv.gz",
    parse_dates=["dod"],
    dtype={"subject_id": "int32", "anchor_age": "int16", "anchor_year": "int16"},
)

cohort_adm = cohort_adm.merge(adm, on="hadm_id", how="left")
cohort_adm = cohort_adm.merge(pts, on="subject_id", how="left")

cohort_adm["age_at_admission"] = cohort_adm["anchor_age"] + \
    (cohort_adm["admittime"].dt.year - cohort_adm["anchor_year"])
cohort_adm["los_days"] = (cohort_adm["dischtime"] - cohort_adm["admittime"]).dt.total_seconds() / 86400

# post-discharge mortality (only computable for those discharged alive, dod known)
alive_disch = cohort_adm["hospital_expire_flag"] == 0
cohort_adm["days_disch_to_death"] = np.where(
    alive_disch & cohort_adm["dod"].notna(),
    (cohort_adm["dod"] - cohort_adm["dischtime"]).dt.total_seconds() / 86400,
    np.nan,
)
cohort_adm["died_30d_postdisch"] = alive_disch & (cohort_adm["days_disch_to_death"] <= 30)
cohort_adm["died_90d_postdisch"] = alive_disch & (cohort_adm["days_disch_to_death"] <= 90)
cohort_adm["died_1y_postdisch"] = alive_disch & (cohort_adm["days_disch_to_death"] <= 365)
cohort_adm["died_inhosp_or_1y"] = (cohort_adm["hospital_expire_flag"] == 1) | (cohort_adm["died_1y_postdisch"])

log(f"\nAdmission-level cohort after merge: {len(cohort_adm):,} rows, "
    f"{cohort_adm['subject_id'].nunique():,} unique patients")
log(f"Missing hadm_id merge to admissions: {cohort_adm['admittime'].isna().sum()}")

# ---------------------------------------------------------------
# 7. Patient-level cohort: first qualifying (index) admission per subject
# ---------------------------------------------------------------
cohort_adm_sorted = cohort_adm.sort_values(["subject_id", "admittime"])
cohort_pts = cohort_adm_sorted.groupby("subject_id", as_index=False).first()
log(f"\nPatient-level cohort (index admission only): {len(cohort_pts):,} patients")

# ---------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------
cohort_adm.to_parquet(f"{DATA_DIR}/diabetic_foot_cohort_admissions.parquet", index=False)
cohort_pts.to_parquet(f"{DATA_DIR}/diabetic_foot_cohort_patients.parquet", index=False)
log(f"\nSaved: data/diabetic_foot_cohort_admissions.parquet ({len(cohort_adm):,} rows)")
log(f"Saved: data/diabetic_foot_cohort_patients.parquet ({len(cohort_pts):,} rows)")

with open(f"{LOG_DIR}/00a_dfu_cohort.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
