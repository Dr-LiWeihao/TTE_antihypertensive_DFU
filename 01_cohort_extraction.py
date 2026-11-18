"""
01_cohort_extraction.py
------------------------
REBUILT 2026-08-31 (CLAUDE.md REVISION 3 -- clone-censor-weight redesign).

Eligibility per the locked protocol:
  - Diabetic foot ulcer/gangrene (reuses project_009's cohort exactly)
  - ANY qualifying admission per patient (NOT restricted to first --
    sequential/repeated-eligible-period trial emulation)
  - Documented hypertension diagnosis (ICD-9 401.x; ICD-10 I10-I13, I15)

REVISION 3 CHANGE: the previous "alive and still admitted at 48h" rule is
REMOVED from eligibility. That rule conditioned study entry on surviving
to a post-baseline reassessment point -- exactly the "Misaligned T0 ->
Selection Bias" pattern in Table 3 of Fourati et al. 2026 (A practical
guide to target trial emulation). Verified against our own data: it was
silently dropping 495 admissions (7 of them deaths) from the eligible
population entirely. See notes/2026-08-31_ccw_redesign_decision.md.

Time zero (T0) is now hospital admission (admittime) -- the actual
clinical decision point. The 48h window becomes a GRACE PERIOD used only
downstream (script 02 exposure ascertainment, script 05 CCW censoring),
not an eligibility filter.

Because a patient can contribute multiple admissions, downstream
inference MUST use patient-level clustered bootstrap resampling, not
admission-level.

Output: data/eligible_cohort.parquet
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("TTE cohort extraction: ACEI/ARB vs CCB/beta-blocker/diuretic in")
log("diabetic foot + hypertension (REBUILT 2026-08-31, REVISION 3 -- CCW)")
log("=" * 70)

# Reuse project_009's diabetic foot cohort exactly -- do not re-derive
adm = pd.read_parquet(data("diabetic_foot_cohort_admissions.parquet"))
log(f"Diabetic foot admissions (project_009 cohort, reused as-is): {len(adm):,}")
log(f"  ({adm.subject_id.nunique():,} unique patients)")

# ALL qualifying admissions per patient (not just first)
idx = adm.copy()

# Hypertension diagnosis (any position, this admission)
diag = pd.read_csv(
    mimic("hosp", "diagnoses_icd"),
    dtype={"subject_id": "int32", "hadm_id": "int32", "icd_code": str, "icd_version": "int8"},
)
diag["icd_code"] = diag["icd_code"].str.strip()
diag = diag[diag.hadm_id.isin(set(idx.hadm_id))]
htn9 = set(diag.loc[(diag.icd_version == 9) & diag.icd_code.str.startswith(HTN_ICD9_PREFIXES), "hadm_id"])
htn10 = set(diag.loc[(diag.icd_version == 10) & diag.icd_code.str.startswith(HTN_ICD10_PREFIXES), "hadm_id"])
htn_hadm = htn9 | htn10

idx["hypertension_dx"] = idx.hadm_id.isin(htn_hadm)
eligible = idx[idx.hypertension_dx].copy()
log(f"Admissions with documented hypertension diagnosis: {len(eligible):,}")
log(f"  ({eligible.subject_id.nunique():,} unique patients)")

# REVISION 3: T0 = admittime, no survival-to-landmark eligibility filter.
eligible["time_zero"] = eligible["admittime"]
eligible["los_hours"] = (eligible["dischtime"] - eligible["admittime"]).dt.total_seconds() / 3600

# Reference-only diagnostic (NOT used to exclude anyone): how many of
# these admissions would have failed the old 48h-survival rule, and how
# many of those were deaths -- quantifies exactly what REVISION 3 recovers.
would_have_failed_48h = eligible["los_hours"] < GRACE_HOURS_PRIMARY
n_recovered = would_have_failed_48h.sum()
n_recovered_died = eligible.loc[would_have_failed_48h, "hospital_expire_flag"].sum()
log(f"[REVISION 3 diagnostic] Admissions that the OLD 48h-landmark "
    f"eligibility rule would have excluded, now retained: {n_recovered:,} "
    f"(of which died in-hospital: {n_recovered_died:,})")

log(f"\nEligible cohort (admission-anchored T0, REVISION 3): {len(eligible):,}")
log(f"  ({eligible.subject_id.nunique():,} unique patients)")
n_multi = eligible.groupby("subject_id").size()
log(f"Patients contributing >1 eligible admission: {(n_multi > 1).sum():,} / {len(n_multi):,} "
    f"({100*(n_multi > 1).mean():.1f}%) -- downstream inference must use patient-level "
    f"clustered bootstrap, not admission-level")

assert eligible["hadm_id"].is_unique, "Expected one row per admission"

eligible.to_parquet(data("eligible_cohort.parquet"), index=False)
log(f"\nSaved: data/eligible_cohort.parquet ({len(eligible):,} rows)")

os.makedirs(LOG_DIR, exist_ok=True)
with open(log_path("01_cohort_extraction.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
