"""
Comorbidity flags + Charlson Comorbidity Index (Quan 2005 ICD-9/10 mapping,
restricted to the 8 categories the reference paper reports plus sepsis, and
the standard 17-category weighted CCI).
"""
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, DATA_DIR, LOG_DIR
BASE = BASE_DIR

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

# Quan (2005) ICD-9-CM and ICD-10 prefix maps, weight per Charlson/Deyo.
CCI_ICD9 = {
    "mi": (["410", "412"], 1),
    "chf": (["428"], 1),
    "peripheral_vascular": (["4439", "4471", "5571", "5579", "V434"], 1),
    "cerebrovascular": (["430", "431", "432", "433", "434", "435", "436", "437", "438"], 1),
    "dementia": (["290"], 1),
    "chronic_pulmonary": (["4168", "4169", "490", "491", "492", "493", "494", "495", "496", "500", "501", "502", "503", "504", "505"], 1),
    "rheumatic": (["7100", "7101", "7104", "7140", "7141", "7142", "725"], 1),
    "mild_liver": (["5712", "5714", "5715", "5716"], 1),
    "diabetes_uncomplicated": (["2500", "2501", "2502", "2503"], 1),
    "diabetes_complicated": (["2504", "2505", "2506", "2507"], 2),
    "hemiplegia": (["342", "3441"], 2),
    "renal": (["582", "583", "585", "586", "V420", "V451"], 2),
    "malignancy": (["14", "15", "16", "170", "171", "172", "174", "175", "176", "179", "18", "190", "191", "192", "193", "194", "1950", "1951", "1952", "1953", "1954", "1955", "1958", "20", "2386"], 2),
    "severe_liver": (["4560", "4561", "4562", "5722", "5723", "5724", "5728"], 3),
    "metastatic": (["196", "197", "198", "199"], 6),
    "aids": (["042", "043", "044"], 6),
}
CCI_ICD10 = {
    "mi": (["I21", "I22", "I252"], 1),
    "chf": (["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429", "I43", "I50", "P290"], 1),
    "peripheral_vascular": (["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"], 1),
    "cerebrovascular": (["G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"], 1),
    "dementia": (["F00", "F01", "F02", "F03", "F051", "G30", "G311"], 1),
    "chronic_pulmonary": (["I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703"], 1),
    "rheumatic": (["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"], 1),
    "mild_liver": (["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K73", "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"], 1),
    "diabetes_uncomplicated": (["E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E120", "E121", "E126", "E128", "E129", "E130", "E131", "E136", "E138", "E139", "E140", "E141", "E146", "E148", "E149"], 1),
    "diabetes_complicated": (["E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115", "E117", "E122", "E123", "E124", "E125", "E127", "E132", "E133", "E134", "E135", "E137", "E142", "E143", "E144", "E145", "E147"], 2),
    "hemiplegia": (["G041", "G114", "G80", "G81", "G82", "G830", "G831", "G832", "G833", "G834", "G839"], 2),
    "renal": (["I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037", "N052", "N053", "N054", "N055", "N056", "N057", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"], 2),
    "malignancy": (["C0", "C1", "C2", "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40", "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C6", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83", "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96", "C97"], 2),
    "severe_liver": (["I850", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"], 3),
    "metastatic": (["C77", "C78", "C79", "C80"], 6),
    "aids": (["B20", "B21", "B22", "B24"], 6),
}
SEPSIS_ICD9 = ["0380", "0381", "03811", "03812", "03819", "0382", "0383", "03840", "03841", "03842", "03843", "03844", "03849", "0388", "0389", "9959"]
SEPSIS_ICD10 = ["A40", "A41", "R6520", "R6521"]

def code_matches(icd_code, prefixes):
    return any(icd_code.startswith(p) for p in prefixes)

adm = pd.read_parquet(f"{DATA_DIR}/diabetic_foot_cohort_admissions.parquet")
diag = pd.read_csv(
    f"{BASE}/hosp/diagnoses_icd.csv.gz",
    dtype={"subject_id": "int32", "hadm_id": "int32", "icd_code": str, "icd_version": "int8"},
)
diag["icd_code"] = diag["icd_code"].str.strip()
hadm_set = set(adm.hadm_id)
diag = diag[diag.hadm_id.isin(hadm_set)].copy()

diag9 = diag[diag.icd_version == 9]
diag10 = diag[diag.icd_version == 10]

result = pd.DataFrame({"hadm_id": sorted(hadm_set)}).set_index("hadm_id")
weights_present = pd.DataFrame(index=result.index)

for cat in CCI_ICD9:
    prefixes9, weight = CCI_ICD9[cat]
    prefixes10, _ = CCI_ICD10[cat]
    hadm9 = set(diag9[diag9.icd_code.apply(lambda c: code_matches(c, prefixes9))].hadm_id)
    hadm10 = set(diag10[diag10.icd_code.apply(lambda c: code_matches(c, prefixes10))].hadm_id)
    flagged = hadm9 | hadm10
    result[cat] = result.index.isin(flagged)
    weights_present[cat] = result[cat].astype(int) * weight

# diabetes: use the higher of uncomplicated/complicated weight, not both (Charlson convention)
diab_weight = weights_present[["diabetes_uncomplicated", "diabetes_complicated"]].max(axis=1)
other_cats = [c for c in CCI_ICD9 if c not in ("diabetes_uncomplicated", "diabetes_complicated")]
result["charlson_comorbidity_index"] = weights_present[other_cats].sum(axis=1) + diab_weight

hadm9_sepsis = set(diag9[diag9.icd_code.apply(lambda c: code_matches(c, SEPSIS_ICD9))].hadm_id)
hadm10_sepsis = set(diag10[diag10.icd_code.apply(lambda c: code_matches(c, SEPSIS_ICD10))].hadm_id)
result["sepsis"] = result.index.isin(hadm9_sepsis | hadm10_sepsis)

keep_cols = ["mi", "chf", "cerebrovascular", "chronic_pulmonary", "mild_liver", "severe_liver",
             "renal", "sepsis", "charlson_comorbidity_index"]
result = result[keep_cols].reset_index()

log(f"Admissions: {len(result):,}")
log(f"CCI distribution:\n{result.charlson_comorbidity_index.describe().to_string()}")
for c in keep_cols[:-1]:
    log(f"{c}: {result[c].sum():,} ({100*result[c].mean():.1f}%)")

result.to_parquet(f"{DATA_DIR}/comorbidity_cci.parquet", index=False)
log("Saved: data/comorbidity_cci.parquet")

with open(f"{LOG_DIR}/00b_comorbidity_cci.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
