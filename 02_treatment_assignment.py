"""
02_treatment_assignment.py
----------------------------
REBUILT 2026-08-31 (REVISION 3). Exposure ascertainment: antihypertensive
regimen observed within a grace period from admission (T0). Computed at
all three grace windows (24h/48h/72h) in one pass over prescriptions, so
scripts 05/06 (primary, grace=48h) and 07 (sensitivity, grace=24h/72h)
share this output without rescanning the 500-800MB prescriptions table
three times.

ANCHOR-DRUG DEFINITION (CLAUDE.md REVISION 2, unchanged by REVISION 3):
  Arm A (ACEI/ARB strategy) = ACEI/ARB present in the grace window,
    REGARDLESS of what else is present.
  Arm B_X (comparator strategy) = drug X present, ACEI/ARB NOT present.

This script only ascertains WHICH drugs were observed in each window --
it does not resolve final "arm" membership. Arm resolution (per grace
window) and the CCW clone/censor step happen in script 05, since REVISION
3 requires both clones to remain at risk for the shared pre-resolution
period (see notes/2026-08-31_ccw_redesign_decision.md).

Output: data/drug_exposure.parquet (one row per eligible admission, with
        aceiarb/ccb/betablocker/diuretic boolean flags at each grace
        window: *_24h, *_48h, *_72h)
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

GRACE_WINDOWS = [24, GRACE_HOURS_PRIMARY, 72]  # union with sensitivity windows, de-dup below
GRACE_WINDOWS = sorted(set(GRACE_WINDOWS))

eligible = pd.read_parquet(data("eligible_cohort.parquet"))
hadm_set = set(eligible.hadm_id)
admittime_lookup = eligible.set_index("hadm_id")["admittime"]

presc = pd.read_csv(
    mimic("hosp", "prescriptions"),
    usecols=["hadm_id", "drug", "starttime"],
    dtype={"hadm_id": "float64", "drug": str},  # float64, NOT float32: hadm_id (~2e7-3e7) exceeds float32 exact-integer range (2^24) -> odd IDs rounded (bug fixed 2026-10-01)
    parse_dates=["starttime"],
)
presc = presc[presc.hadm_id.notna()]
presc["hadm_id"] = presc["hadm_id"].astype("int32")
presc = presc[presc.hadm_id.isin(hadm_set)]
presc["admittime"] = presc["hadm_id"].map(admittime_lookup)
presc["hours_from_admit"] = (presc["starttime"] - presc["admittime"]).dt.total_seconds() / 3600
# Orders before admission (rare, e.g. clock/reconciliation artifacts) are
# not evidence of an in-admission regimen decision -- require >= 0.
presc = presc[presc["hours_from_admit"] >= 0]

drug_lower = presc["drug"].str.lower().fillna("")
DRUG_CLASSES = {
    "aceiarb": ACEI_ARB_DRUGS,
    "ccb": CCB_DRUGS,
    "betablocker": BETABLOCKER_DRUGS,
    "diuretic": DIURETIC_DRUGS,
    # Subclasses for the comparator-definition sensitivity analysis (script 10,
    # added 2026-11-10): dihydropyridine CCB; thiazide-type and loop diuretics.
    "ccbdhp": CCB_DHP_DRUGS,
    "thiazide": THIAZIDE_DRUGS,
    "loop": LOOP_DRUGS,
}
class_mask = {name: drug_lower.str.contains("|".join(drugs)) for name, drugs in DRUG_CLASSES.items()}

out = eligible[["hadm_id", "subject_id", "admittime"]].copy()
for cls_name, mask in class_mask.items():
    cls_presc = presc[mask]
    for gh in GRACE_WINDOWS:
        hadm_in_window = set(cls_presc.loc[cls_presc["hours_from_admit"] <= gh, "hadm_id"])
        out[f"{cls_name}_{gh}h"] = out["hadm_id"].isin(hadm_in_window)
# Restricted diuretic strategy (sensitivity): thiazide-type diuretic WITHOUT a loop
# diuretic in the window, i.e. an antihypertensive rather than a decongestive regimen.
for gh in GRACE_WINDOWS:
    out[f"thiazidenoloop_{gh}h"] = out[f"thiazide_{gh}h"] & ~out[f"loop_{gh}h"]

# ADDED 2026-11-10 (notes/2026-11-10_grace_deviation_fix.md): first order time (hours from
# admission) of the drugs whose order makes a comparator clone deviate, so that script 05 can
# censor a comparator clone that deviated BEFORE an event inside the grace period.
FIRST_ORDER = {"aceiarb": ACEI_ARB_DRUGS, "loop": LOOP_DRUGS,
               "bbvaso": VASODILATING_BB_DRUGS, "bbnonvaso": NONVASODILATING_BB_DRUGS}
for name, drugs in FIRST_ORDER.items():
    first_h = presc.loc[drug_lower.str.contains("|".join(drugs)), ["hadm_id", "hours_from_admit"]] \
        .groupby("hadm_id")["hours_from_admit"].min()
    out[f"{name}_first_h"] = out["hadm_id"].map(first_h)
for gh in GRACE_WINDOWS:
    assert ((out["aceiarb_first_h"] <= gh) == out[f"aceiarb_{gh}h"]).all(), "first-order time disagrees with flags"
    assert ((out["loop_first_h"] <= gh) == out[f"loop_{gh}h"]).all(), "first-order time disagrees with flags"

arni = presc.loc[drug_lower.str.contains("sacubitril|entresto") & (presc["hours_from_admit"] <= GRACE_HOURS_PRIMARY), "hadm_id"].nunique()
log(f"Sacubitril/valsartan within {GRACE_HOURS_PRIMARY}h (counted as ACEI/ARB via valsartan): {arni}")

log(f"Eligible admissions: {len(out):,}")
for cls_name in DRUG_CLASSES:
    counts = ", ".join(f"{gh}h={out[f'{cls_name}_{gh}h'].sum():,}" for gh in GRACE_WINDOWS)
    log(f"{cls_name}: {counts}")

# Primary-window overlap diagnostic (matches REVISION 2's original check)
gh = GRACE_HOURS_PRIMARY
aceiarb_col = f"aceiarb_{gh}h"
log(f"\nACEI/ARB present at primary grace window ({gh}h): {out[aceiarb_col].sum():,}")
for name in ["ccb", "betablocker", "diuretic"]:
    col = f"{name}_{gh}h"
    overlap = (out[aceiarb_col] & out[col]).sum()
    pct = 100 * overlap / out[aceiarb_col].sum()
    log(f"  Also on {name} ({gh}h): {overlap:,} ({pct:.1f}% of ACEI/ARB-exposed)")

out.to_parquet(data("drug_exposure.parquet"), index=False)
log("\nSaved: data/drug_exposure.parquet")

with open(log_path("02_treatment_assignment.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
