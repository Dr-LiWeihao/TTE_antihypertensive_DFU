"""
16_extract_review3.py
---------------------
NEW 2026-11-13. Extraction for the review-v3 Tier 2 analyses, pre-specified BEFORE running in
notes/2026-11-13_review3_tier2_prespec.md. Primary analysis untouched.

Per eligible admission (data/review3_covariates.parquet):
  ed_linked, ed_sbp, ed_dbp            ED triage BP of the ED stay linked to the admission (MIMIC-IV-ED 2.2)
  medrecon_available, pre_<class>      ED medication reconciliation lists the class (aceiarb/ccb/betablocker/diuretic)
  ccb_first_h, betablocker_first_h, diuretic_first_h   first order time (h from admission), symmetric comparator
  last24_<class>                       an order of the class active in the last 24 h of the index stay
  aceiarb_after48h                     first ACEI/ARB order after 48 h
  early_revasc, early_amp, early_debrid, early_limb_proc   limb procedure on admission day or next day
  index_revasc                         any lower-limb revascularization during the index admission
  major_amp_8410_only                  first major amputation coded only as ICD-9 84.10
Output also: results/tables/review3_procedure_codes.csv (code lists, for Table S8)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *

log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

# ED_DIR (MIMIC-IV-ED v2.2 ed/ folder) comes from config.py
G = GRACE_HOURS_PRIMARY
CLASSES = {"aceiarb": ACEI_ARB_DRUGS, "ccb": CCB_DRUGS, "betablocker": BETABLOCKER_DRUGS, "diuretic": DIURETIC_DRUGS}

el = pd.read_parquet(data("eligible_cohort.parquet"))
ids = set(el.hadm_id)
res = el[["hadm_id", "subject_id", "admittime", "dischtime"]].copy()

# ---------------- ED triage and medication reconciliation ----------------
eds = pd.read_csv(os.path.join(ED_DIR, "edstays.csv.gz"), usecols=["subject_id", "hadm_id", "stay_id", "intime"],
                  dtype={"subject_id": "int64", "hadm_id": "Int64", "stay_id": "int64"}, parse_dates=["intime"])
eds = eds[eds.hadm_id.notna()].copy()
eds["hadm_id"] = eds["hadm_id"].astype("int64")
eds = eds[eds.hadm_id.isin(ids)].merge(el[["hadm_id", "admittime"]], on="hadm_id")
eds = eds[eds.intime <= eds.admittime + pd.Timedelta(hours=1)]
eds = eds.sort_values("intime").groupby("hadm_id").last().reset_index()
log(f"Admissions with a linked ED stay: {len(eds):,} / {len(res):,} ({100*len(eds)/len(res):.1f}%)")
tri = pd.read_csv(os.path.join(ED_DIR, "triage.csv.gz"), usecols=["stay_id", "sbp", "dbp"], dtype={"stay_id": "int64"})
eds = eds.merge(tri, on="stay_id", how="left")
eds["sbp"] = eds["sbp"].where(eds["sbp"].between(50, 300))
eds["dbp"] = eds["dbp"].where(eds["dbp"].between(20, 200))
med = pd.read_csv(os.path.join(ED_DIR, "medrecon.csv.gz"), usecols=["stay_id", "name"], dtype={"stay_id": "int64", "name": str})
med = med[med.stay_id.isin(set(eds.stay_id))]
med["n"] = med["name"].str.lower().fillna("")
mr = pd.DataFrame({"stay_id": sorted(set(med.stay_id))})
for cls, drugs in CLASSES.items():
    mr[f"pre_{cls}"] = mr.stay_id.isin(set(med.loc[med.n.str.contains("|".join(drugs)), "stay_id"])).astype(int)
mr["medrecon_available"] = 1
eds = eds.merge(mr, on="stay_id", how="left")
res = res.merge(eds[["hadm_id", "sbp", "dbp", "medrecon_available"] + [f"pre_{c}" for c in CLASSES]]
                .rename(columns={"sbp": "ed_sbp", "dbp": "ed_dbp"}), on="hadm_id", how="left")
res["ed_linked"] = res.hadm_id.isin(set(eds.hadm_id)).astype(int)
res["medrecon_available"] = res["medrecon_available"].fillna(0).astype(int)
log(f"ED triage SBP available: {res.ed_sbp.notna().sum():,} ({100*res.ed_sbp.notna().mean():.1f}%); "
    f"medrecon available: {res.medrecon_available.sum():,} ({100*res.medrecon_available.mean():.1f}%)")
for c in CLASSES:
    m = res.medrecon_available == 1
    log(f"  pre-admission {c} (medrecon, among available): {100*res.loc[m, f'pre_{c}'].mean():.1f}%")

# ---------------- Prescriptions: first-order times, persistence ----------------
adm = el.set_index("hadm_id")["admittime"]
dis = el.set_index("hadm_id")["dischtime"]
pat_all = "|".join(sum(CLASSES.values(), []))
parts = []
for ch in pd.read_csv(mimic("hosp", "prescriptions"), usecols=["hadm_id", "drug", "starttime", "stoptime"],
                      dtype={"hadm_id": "float64", "drug": str}, parse_dates=["starttime", "stoptime"],
                      chunksize=2_000_000):
    ch = ch[ch.hadm_id.notna()].copy()
    ch["hadm_id"] = ch["hadm_id"].astype("int64")   # float64 -> int64, never float32
    ch = ch[ch.hadm_id.isin(ids)]
    ch["dl"] = ch["drug"].str.lower().fillna("")
    parts.append(ch[ch.dl.str.contains(pat_all)])
pr = pd.concat(parts, ignore_index=True)
pr["h"] = (pr["starttime"] - pr["hadm_id"].map(adm)).dt.total_seconds() / 3600
pr = pr[pr.h >= 0]
disc = pr["hadm_id"].map(dis)
active_last24 = (pr["starttime"] < disc) & (pr["stoptime"].isna() | (pr["stoptime"] > disc - pd.Timedelta(hours=24)))
drugs02 = pd.read_parquet(data("drug_exposure.parquet")).set_index("hadm_id")
for cls, dlist in CLASSES.items():
    m = pr.dl.str.contains("|".join(dlist))
    first_h = pr.loc[m].groupby("hadm_id")["h"].min()
    res[f"last24_{cls}"] = res.hadm_id.isin(set(pr.loc[m & active_last24, "hadm_id"])).astype(int)
    if cls != "aceiarb":
        res[f"{cls}_first_h"] = res.hadm_id.map(first_h)
        chk = (res.set_index("hadm_id")[f"{cls}_first_h"] <= G) == drugs02.loc[res.hadm_id, f"{cls}_{G}h"].values
        assert chk.all(), f"{cls}: first-order time disagrees with the script-02 {G}-h flag"
    else:
        ever_after = set(pr.loc[m & (pr.h > G), "hadm_id"])
        res["aceiarb_after48h"] = res.hadm_id.isin(ever_after).astype(int)
log("First-order times for ccb/betablocker/diuretic reproduce the script-02 48-h flags (checked).")

# ---------------- Procedures ----------------
dproc = pd.read_csv(mimic("hosp", "d_icd_procedures"), dtype=str)
d10 = dproc[dproc.icd_version == "10"]
ART = r"(?:External Iliac|Common Iliac|Femoral|Popliteal|Anterior Tibial|Posterior Tibial|Peroneal|Foot) Arter|Lower Artery"
REVASC10 = set(d10.loc[d10.long_title.str.match(r"^(?:Bypass|Dilation|Extirpation of Matter from) ")
                       & d10.long_title.str.contains(ART), "icd_code"])
SITE = r"(?:Lower Leg|Foot|Toe|Ankle|Tarsal|Metatarsal)"
DEBRID10 = set(d10.loc[d10.long_title.str.match(r"^Excision of ") & d10.long_title.str.contains(SITE)
                       & d10.icd_code.str[1].isin(list("HJKLMQ")), "icd_code"])
REVASC9 = ["3808", "3818", "3929", "3950", "3990", "0055"]
DEBRID9 = ["8622", "7767", "7768", "7769"]
codes_out = []
for name, c9, c10 in [("Lower-limb revascularization", REVASC9, REVASC10), ("Surgical debridement or excision, lower leg or foot", DEBRID9, DEBRID10)]:
    codes_out.append({"group": name, "icd9": ", ".join(c9), "icd10_pcs_n": len(c10),
                      "icd10_pcs_prefixes": ", ".join(sorted({c[:4] for c in c10}))})
pd.DataFrame(codes_out).to_csv(results_tables("review3_procedure_codes.csv"), index=False)
log(f"Revascularization ICD-10-PCS codes: {len(REVASC10)}; debridement/excision codes: {len(DEBRID10)}")

proc = pd.read_csv(mimic("hosp", "procedures_icd"), usecols=["hadm_id", "chartdate", "icd_code", "icd_version"],
                   parse_dates=["chartdate"], dtype={"hadm_id": "int64", "icd_code": str, "icd_version": "int8"})
proc["icd_code"] = proc["icd_code"].str.strip()
pidx = proc[proc.hadm_id.isin(ids)].copy()
v9, v10 = pidx.icd_version == 9, pidx.icd_version == 10
c = pidx.icd_code
is_rev = (v9 & c.isin(REVASC9)) | (v10 & c.isin(REVASC10))
is_deb = (v9 & c.isin(DEBRID9)) | (v10 & c.isin(DEBRID10))
is_amp = (v9 & c.isin(MAJOR_AMP_ICD9 + OTHER_AMP_ICD9)) | (v10 & c.str.startswith(MAJOR_AMP_ICD10_PREFIXES + OTHER_AMP_ICD10_PREFIXES))
day = (pidx["chartdate"] - pidx["hadm_id"].map(adm).dt.normalize()).dt.days
early = day <= 1
res["early_revasc"] = res.hadm_id.isin(set(pidx.loc[is_rev & early, "hadm_id"])).astype(int)
res["early_amp"] = res.hadm_id.isin(set(pidx.loc[is_amp & early, "hadm_id"])).astype(int)
res["early_debrid"] = res.hadm_id.isin(set(pidx.loc[is_deb & early, "hadm_id"])).astype(int)
res["early_limb_proc"] = res[["early_revasc", "early_amp", "early_debrid"]].max(axis=1)
res["index_revasc"] = res.hadm_id.isin(set(pidx.loc[is_rev, "hadm_id"])).astype(int)
for v in ["early_revasc", "early_amp", "early_debrid", "early_limb_proc", "index_revasc"]:
    log(f"{v}: {res[v].sum():,} ({100*res[v].mean():.1f}%)")

# ---------------- First major amputation coded only as 84.10 ----------------
out = pd.read_parquet(data("outcomes.parquet"))
ev = out[out.amputation_only_event.astype(bool)][["hadm_id", "admittime", "amp_hadm_id", "days_to_major_amp"]].copy()
ev["amp_date"] = (ev["admittime"] + pd.to_timedelta(ev["days_to_major_amp"], unit="D")).dt.round("D")
pall = proc[proc.hadm_id.isin(set(ev.amp_hadm_id.dropna().astype("int64")))]
is_major = ((pall.icd_version == 9) & pall.icd_code.isin(MAJOR_AMP_ICD9)) | \
           ((pall.icd_version == 10) & pall.icd_code.str.startswith(MAJOR_AMP_ICD10_PREFIXES))
pm = pall[is_major]
codes_by = pm.groupby(["hadm_id", "chartdate"])["icd_code"].apply(set)
def only_8410(r):
    s = codes_by.get((int(r.amp_hadm_id), r.amp_date))
    return s is not None and s == {"8410"}
ev["only8410"] = ev.apply(only_8410, axis=1)
matched = ev.apply(lambda r: (int(r.amp_hadm_id), r.amp_date) in codes_by.index, axis=1)
log(f"First major amputations (AFS-eligible, after day 0): {len(ev):,}; matched to a coded date: {matched.sum():,}; "
    f"coded only as 84.10: {ev.only8410.sum():,}")
res["major_amp_8410_only"] = res.hadm_id.isin(set(ev.loc[ev.only8410, "hadm_id"])).astype(int)

res = res.drop(columns=["admittime", "dischtime"])
res.to_parquet(data("review3_covariates.parquet"), index=False)
log(f"Saved data/review3_covariates.parquet ({res.shape})")
with open(log_path("16_extract_review3.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
