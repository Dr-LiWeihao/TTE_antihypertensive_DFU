"""
11_extract_early_labs_icu.py
-----------------------------
NEW 2026-11-10 (second simulated review; pre-specified in
notes/2026-11-10_review2_sensitivity_prespec.md). Extra admission-level variables for
SENSITIVITY analyses only (script 12); the primary analysis does not use them.

  first_potassium_24h, first_creatinine_24h  first value charted at/after admittime and <=24 h
                                              (A1b baseline-contraindication restriction)
  contraindication_ras                        first K > 5.5 mmol/L or first creatinine > 3.0 mg/dL
  crea_change_48h, max_potassium_48h          0-48 h creatinine change (max - first) and max K
  crea_change_48h_unknown, max_potassium_48h_unknown   missingness indicators   (B7)
  icu_within_48h                              ICU intime <= admittime + 48 h   (B5)
  era_*                                       approximate admission calendar era (B6)
  later_adm_365d                              >=1 later MIMIC-IV admission within 365 d after
                                              index discharge                  (A2a)

IDs read as int64/float64, never float32 (CLAUDE.md hard rule).
Output: data/sensitivity_covariates.parquet
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

elig = pd.read_parquet(data("eligible_cohort.parquet"))[
    ["hadm_id", "subject_id", "admittime", "dischtime", "anchor_year", "anchor_year_group", "hospital_expire_flag"]]
hadm_set = set(elig.hadm_id.astype("int64"))
log(f"Eligible admissions: {len(elig):,}")

# --- Labs: creatinine (50912, 52546) and potassium (50971, 50822) -------------------
CREA, POT = [50912, 52546], [50971, 50822]
rows = []
for ch in pd.read_csv(mimic("hosp", "labevents"), usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                      dtype={"hadm_id": "float64", "itemid": "int32"}, parse_dates=["charttime"],
                      chunksize=3_000_000):
    ch = ch[ch.hadm_id.notna() & ch.itemid.isin(CREA + POT) & ch.valuenum.notna()]
    if ch.empty:
        continue
    ch["hadm_id"] = ch["hadm_id"].astype("int64")
    ch = ch[ch.hadm_id.isin(hadm_set)]
    if not ch.empty:
        rows.append(ch)
lab = pd.concat(rows, ignore_index=True)
lab = lab.merge(elig[["hadm_id", "admittime"]], on="hadm_id")
lab["h"] = (lab["charttime"] - lab["admittime"]).dt.total_seconds() / 3600
lab = lab[(lab.h >= 0) & (lab.h <= 48)]
lab["analyte"] = np.where(lab.itemid.isin(CREA), "crea", "k")
log(f"Lab rows within 0-48 h: {len(lab):,} (creatinine {(lab.analyte=='crea').sum():,}, "
    f"potassium {(lab.analyte=='k').sum():,})")

lab = lab.sort_values(["hadm_id", "charttime"])
first24 = lab[lab.h <= 24].groupby(["hadm_id", "analyte"])["valuenum"].first().unstack()
crea48 = lab[lab.analyte == "crea"].groupby("hadm_id")["valuenum"].agg(["first", "max"])
k48 = lab[lab.analyte == "k"].groupby("hadm_id")["valuenum"].max()

out = elig[["hadm_id", "subject_id"]].copy().set_index("hadm_id")
out["first_creatinine_24h"] = first24.get("crea")
out["first_potassium_24h"] = first24.get("k")
out["contraindication_ras"] = ((out["first_potassium_24h"] > 5.5) | (out["first_creatinine_24h"] > 3.0)).astype(int)
out["crea_change_48h"] = crea48["max"] - crea48["first"]
out["max_potassium_48h"] = k48
for c in ["crea_change_48h", "max_potassium_48h"]:
    out[f"{c}_unknown"] = out[c].isna().astype(int)
log(f"First K (<=24 h) available: {out.first_potassium_24h.notna().mean():.1%}; "
    f"first creatinine (<=24 h): {out.first_creatinine_24h.notna().mean():.1%}")
log(f"Baseline contraindication (K>5.5 or creatinine>3.0): {out.contraindication_ras.sum():,} "
    f"({out.contraindication_ras.mean():.1%})")
log(f"0-48 h creatinine change available {1 - out.crea_change_48h_unknown.mean():.1%}, "
    f"median {out.crea_change_48h.median():.2f}; max K available {1 - out.max_potassium_48h_unknown.mean():.1%}")

# --- ICU within 48 h of admission -----------------------------------------------------
icu = pd.read_csv(mimic("icu", "icustays"), usecols=["hadm_id", "intime"], dtype={"hadm_id": "int64"},
                  parse_dates=["intime"])
icu = icu.merge(elig[["hadm_id", "admittime"]], on="hadm_id")
icu48 = set(icu.loc[(icu.intime - icu.admittime).dt.total_seconds() <= 48 * 3600, "hadm_id"])
out["icu_within_48h"] = out.index.isin(icu48).astype(int)
log(f"ICU admission within 48 h: {out.icu_within_48h.sum():,} ({out.icu_within_48h.mean():.1%})")

# --- Approximate calendar era -----------------------------------------------------------
e = elig.set_index("hadm_id")
mid = e["anchor_year_group"].str.slice(0, 4).astype(int) + 1
approx_year = e["admittime"].dt.year - e["anchor_year"] + mid
era = pd.cut(approx_year, [-np.inf, 2010, 2013, 2016, 2019, np.inf],
             labels=["le2010", "2011_2013", "2014_2016", "2017_2019", "ge2020"])
log("Approximate admission era:\n" + era.value_counts().sort_index().to_string())
dummies = pd.get_dummies(era, prefix="era", drop_first=True).astype(int)
out = out.join(dummies)

# --- Later same-hospital admission within 365 d after index discharge ---------------
adm = pd.read_csv(mimic("hosp", "admissions"), usecols=["subject_id", "hadm_id", "admittime"],
                  dtype={"subject_id": "int64", "hadm_id": "int64"}, parse_dates=["admittime"])
adm = adm[adm.subject_id.isin(set(elig.subject_id))].rename(columns={"hadm_id": "later_hadm", "admittime": "later_admit"})
m = elig[["hadm_id", "subject_id", "dischtime"]].merge(adm, on="subject_id")
m = m[(m.later_hadm != m.hadm_id) & (m.later_admit > m.dischtime) &
      ((m.later_admit - m.dischtime).dt.total_seconds() <= 365 * 86400)]
out["later_adm_365d"] = out.index.isin(set(m.hadm_id)).astype(int)
alive = e["hospital_expire_flag"] == 0
log(f"Later MIMIC-IV admission within 365 d of discharge (alive at discharge): "
    f"{out.loc[alive[alive].index, 'later_adm_365d'].mean():.1%}")

out.reset_index().to_parquet(data("sensitivity_covariates.parquet"), index=False)
log("Saved: data/sensitivity_covariates.parquet")
with open(log_path("11_extract_early_labs_icu.log"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
