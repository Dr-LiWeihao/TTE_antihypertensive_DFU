"""
14_extract_pad_bbagent_anyamp.py
---------------------------------
NEW 2026-11-10. EXPLORATORY, post hoc (pre-specified before running in
notes/2026-11-10_betablocker_pad_exploration_prespec.md).

Per eligible admission:
  clti                index-admission code for rest pain, ischaemic ulceration or gangrene
  athero_pad          index-admission code for atherosclerosis of the extremity arteries (440.2x / I70.2x)
  bbvaso_48h          vasodilating beta-blocker (carvedilol, labetalol, nebivolol) ordered within 48 h
  bbnonvaso_48h       any other listed beta-blocker ordered within 48 h
  bbvasoonly_48h      bbvaso and not bbnonvaso   (comparator flags for script 15)
  bbnonvasoonly_48h   bbnonvaso and not bbvaso
  anyamp_event / anyamp_days_from_t0 / anyamp_at_t0
                      first major OR minor lower-extremity amputation, ascertained as in script 04
                      (all of the patient's admissions, admission calendar day to end of mortality
                      follow-up; death = censoring; day-0 events flagged for exclusion)

Output: data/pad_bb_anyamp.parquet
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

G = GRACE_HOURS_PRIMARY
VASO_BB, NONVASO_BB = VASODILATING_BB_DRUGS, NONVASODILATING_BB_DRUGS  # config.py
MINOR_AMP_ICD9, MINOR_AMP_ICD10_PREFIXES = OTHER_AMP_ICD9, OTHER_AMP_ICD10_PREFIXES

el = pd.read_parquet(data("eligible_cohort.parquet"))
out = pd.read_parquet(data("outcomes.parquet"))
ids = set(el.hadm_id)
res = el[["hadm_id", "subject_id"]].copy()

# --- diagnosis-based modifiers ---
dx = pd.read_csv(mimic("hosp", "diagnoses_icd"), usecols=["hadm_id", "icd_code", "icd_version"],
                 dtype={"hadm_id": "int64", "icd_code": str, "icd_version": "int8"})
dx = dx[dx.hadm_id.isin(ids)]
c = dx.icd_code.str.strip()
v9, v10 = dx.icd_version == 9, dx.icd_version == 10
clti_m = (v9 & c.isin(CLTI_ICD9)) | (v10 & (c.str.startswith(CLTI_ICD10_PREFIXES) | c.str.match(CLTI_ICD10_REGEX)))
athero_m = (v9 & c.str.startswith(ATHERO_PAD_ICD9_PREFIX)) | (v10 & c.str.startswith(ATHERO_PAD_ICD10_PREFIX))
res["clti"] = res.hadm_id.isin(set(dx.loc[clti_m, "hadm_id"])).astype(int)
res["athero_pad"] = res.hadm_id.isin(set(dx.loc[athero_m, "hadm_id"])).astype(int)
log(f"CLTI: {res.clti.sum():,} ({100*res.clti.mean():.1f}%); atherosclerotic PAD: {res.athero_pad.sum():,} "
    f"({100*res.athero_pad.mean():.1f}%) of {len(res):,}")

# --- beta-blocker subtype within the grace window (same rules as script 02) ---
adm = el.set_index("hadm_id")["admittime"]
pat_all = "|".join(BETABLOCKER_DRUGS)
parts = []
for ch in pd.read_csv(mimic("hosp", "prescriptions"), usecols=["hadm_id", "drug", "starttime"],
                      dtype={"hadm_id": "float64", "drug": str}, parse_dates=["starttime"], chunksize=2_000_000):
    ch = ch[ch.hadm_id.notna()].copy()
    ch["hadm_id"] = ch["hadm_id"].astype("int64")
    ch = ch[ch.hadm_id.isin(ids)]
    dl = ch["drug"].str.lower().fillna("")
    ch = ch[dl.str.contains(pat_all)].copy()
    h = (ch["starttime"] - ch["hadm_id"].map(adm)).dt.total_seconds() / 3600
    parts.append(ch[(h >= 0) & (h <= G)])
bb = pd.concat(parts)
dl = bb["drug"].str.lower()
vaso_ids = set(bb.loc[dl.str.contains("|".join(VASO_BB)), "hadm_id"])
nonvaso_ids = set(bb.loc[dl.str.contains("|".join(NONVASO_BB)), "hadm_id"])
res[f"bbvaso_{G}h"] = res.hadm_id.isin(vaso_ids)
res[f"bbnonvaso_{G}h"] = res.hadm_id.isin(nonvaso_ids)
res[f"bbvasoonly_{G}h"] = res[f"bbvaso_{G}h"] & ~res[f"bbnonvaso_{G}h"]
res[f"bbnonvasoonly_{G}h"] = res[f"bbnonvaso_{G}h"] & ~res[f"bbvaso_{G}h"]
drugs = pd.read_parquet(data("drug_exposure.parquet")).set_index("hadm_id")
chk = res.set_index("hadm_id")
assert ((chk[f"bbvaso_{G}h"] | chk[f"bbnonvaso_{G}h"]) == drugs.loc[chk.index, f"betablocker_{G}h"]).all(), \
    "subtype flags do not reproduce the script-02 beta-blocker flag"
noacei = ~drugs.loc[chk.index, f"aceiarb_{G}h"].values
log(f"Beta-blocker within {G} h, no ACEI/ARB: vasodilating only {(chk[f'bbvasoonly_{G}h'].values & noacei).sum():,}, "
    f"non-vasodilating only {(chk[f'bbnonvasoonly_{G}h'].values & noacei).sum():,}, "
    f"both {(chk[f'bbvaso_{G}h'].values & chk[f'bbnonvaso_{G}h'].values & noacei).sum():,}")

# --- any (major or minor) amputation, same ascertainment as script 04 ---
proc = pd.read_csv(mimic("hosp", "procedures_icd"),
                   usecols=["subject_id", "hadm_id", "chartdate", "icd_code", "icd_version"],
                   parse_dates=["chartdate"],
                   dtype={"subject_id": "int64", "hadm_id": "int64", "icd_version": "int8", "icd_code": str})
proc["icd_code"] = proc["icd_code"].str.strip()
proc = proc[proc.subject_id.isin(set(res.subject_id))]
is_any = ((proc.icd_version == 9) & proc.icd_code.isin(MAJOR_AMP_ICD9 + MINOR_AMP_ICD9)) | \
         ((proc.icd_version == 10) & proc.icd_code.str.startswith(MAJOR_AMP_ICD10_PREFIXES + MINOR_AMP_ICD10_PREFIXES))
amp = proc.loc[is_any, ["subject_id", "hadm_id", "chartdate"]].rename(columns={"hadm_id": "amp_hadm_id"})
tx = out[["hadm_id", "subject_id", "admittime", "mortality_days_from_t0"]]
cand = tx.merge(amp, on="subject_id", how="inner")
cand["d"] = (cand["chartdate"] - cand["admittime"]).dt.total_seconds() / 86400
cand = cand[((cand["chartdate"] >= cand["admittime"].dt.normalize()) | (cand["amp_hadm_id"] == cand["hadm_id"])) &
            (cand["d"] <= cand["mortality_days_from_t0"])]
first = cand.sort_values("d").groupby("hadm_id")["d"].first().clip(lower=0)
tx = tx.set_index("hadm_id")
d_any = tx.index.map(first)
res = res.set_index("hadm_id")
res["anyamp_at_t0"] = pd.Series(d_any, index=tx.index).reindex(res.index).fillna(-1).eq(0)
has = pd.Series(d_any, index=tx.index).reindex(res.index)
res["anyamp_event"] = has.notna() & (has > 0)
res["anyamp_days_from_t0"] = has.where(res["anyamp_event"], tx["mortality_days_from_t0"].reindex(res.index))
res = res.reset_index()
log(f"Any amputation within follow-up: {(res.anyamp_event | res.anyamp_at_t0).sum():,}; day 0 (excluded): "
    f"{res.anyamp_at_t0.sum():,}; after day 0: {res.anyamp_event.sum():,}")
log(f"  among admissions without day-0 any amputation: {res.loc[~res.anyamp_at_t0, 'anyamp_event'].sum():,} / "
    f"{(~res.anyamp_at_t0).sum():,}")

res.to_parquet(data("pad_bb_anyamp.parquet"), index=False)
log("Saved: data/pad_bb_anyamp.parquet")
with open(log_path("14_extract_pad_bbagent_anyamp.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
