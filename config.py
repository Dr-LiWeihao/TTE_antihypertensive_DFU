"""Shared configuration for all analysis scripts."""
import os

# Paths (set by environment variables; see README.md)
#   MIMIC_IV_DIR    folder containing MIMIC-IV v3.1 hosp/ and icu/
#   MIMIC_IV_ED_DIR folder containing the MIMIC-IV-ED v2.2 ed/ tables (script 16 only)
#   OUTPUT_DIR      where derived data, results and logs are written (default ./output)
BASE_DIR = os.environ.get("MIMIC_IV_DIR", "mimic-iv-3.1")
ED_DIR = os.environ.get("MIMIC_IV_ED_DIR", os.path.join("mimic-iv-ed-2.2", "ed"))
OUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
DATA_DIR = os.path.join(OUT_DIR, "data")
RES_DIR = os.path.join(OUT_DIR, "results")
LOG_DIR = os.path.join(OUT_DIR, "logs")
for _d in (DATA_DIR, os.path.join(RES_DIR, "tables"), os.path.join(RES_DIR, "figures"), LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Analysis parameters ---
# REVISION 3 (2026-08-31): T0 is hospital admission (admittime), not a
# 48h landmark -- the landmark is now a GRACE PERIOD for exposure
# ascertainment / CCW censoring only, not an eligibility filter. See
# notes/2026-08-31_ccw_redesign_decision.md.
GRACE_HOURS_PRIMARY = 48
GRACE_HOURS_SENSITIVITY = [24, 72]
N_BOOT = 1000
SEED = 42
BP_LOOKBACK_DAYS = 365   # omr baseline BP window, pre-admission

# --- Hypertension ICD codes ---
HTN_ICD9_PREFIXES = ("401",)
HTN_ICD10_PREFIXES = ("I10", "I11", "I12", "I13", "I15")

# --- Drug classification (drug-name substring match against prescriptions.drug) ---
# COMPLETED 2026-11-10 (pre-submission review, notes/2026-11-10_druglist_amputation_rerun.md).
# The earlier lists (archived in results/_superseded_2026-11-10_pre_druglist_amputation_fix/
# scripts/config.py) missed agents present in this cohort: torsemide (259 admissions),
# bumetanide (40), metolazone (31), quinapril (21), felodipine (17), nadolol (23), ...
# Decisions: sacubitril/valsartan stays in ACEI/ARB (contains an ARB; matched via
# "valsartan"); sotalol excluded (class III antiarrhythmic indication); nimodipine
# excluded (subarachnoid-haemorrhage indication); ophthalmic timolol not listed.
ACEI_DRUGS = ["benazepril", "captopril", "enalapril", "fosinopril", "lisinopril", "moexipril",
              "perindopril", "quinapril", "ramipril", "trandolapril"]
ARB_DRUGS = ["azilsartan", "candesartan", "eprosartan", "irbesartan", "losartan", "olmesartan",
             "telmisartan", "valsartan"]
ACEI_ARB_DRUGS = ACEI_DRUGS + ARB_DRUGS
CCB_DHP_DRUGS = ["amlodipine", "clevidipine", "felodipine", "isradipine", "nicardipine",
                 "nifedipine", "nisoldipine"]
CCB_NONDHP_DRUGS = ["diltiazem", "verapamil"]
CCB_DRUGS = CCB_DHP_DRUGS + CCB_NONDHP_DRUGS
BETABLOCKER_DRUGS = ["acebutolol", "atenolol", "betaxolol", "bisoprolol", "carvedilol", "esmolol",
                     "labetalol", "metoprolol", "nadolol", "nebivolol", "pindolol", "propranolol"]
THIAZIDE_DRUGS = ["chlorothiazide", "chlorthalidone", "hydrochlorothiazide", "indapamide", "metolazone"]
LOOP_DRUGS = ["bumetanide", "ethacrynic", "furosemide", "torsemide"]
MRA_KSPARING_DRUGS = ["amiloride", "eplerenone", "spironolactone", "triamterene"]
DIURETIC_DRUGS = THIAZIDE_DRUGS + LOOP_DRUGS + MRA_KSPARING_DRUGS

# --- Major lower-extremity amputation (above the foot), procedures_icd ---
# COMPLETED 2026-11-10: adds knee/hip disarticulation and hindquarter/femoral-region
# codes to project_009's list (8410, 8414, 8415, 8417; 0Y6C/D/H/J).
# ICD-9-CM: 84.10 lower limb NOS, 84.14 ankle (Syme), 84.15 below knee, 84.16 knee
# disarticulation, 84.17 above knee, 84.18 hip disarticulation, 84.19 abdominopelvic.
# ICD-10-PCS 0Y6 body parts: 2/3/4 hindquarter, 7/8 femoral region, C/D upper leg,
# F/G knee region, H/J lower leg.
MAJOR_AMP_ICD9 = ["8410", "8414", "8415", "8416", "8417", "8418", "8419"]
MAJOR_AMP_ICD10_PREFIXES = ("0Y62", "0Y63", "0Y64", "0Y67", "0Y68", "0Y6C", "0Y6D",
                            "0Y6F", "0Y6G", "0Y6H", "0Y6J")


# --- Subgroup / beta-blocker-type / any-amputation definitions (scripts 14-15, 2026-11-10) ---
VASODILATING_BB_DRUGS = ["carvedilol", "labetalol", "nebivolol"]
NONVASODILATING_BB_DRUGS = [d for d in BETABLOCKER_DRUGS if d not in VASODILATING_BB_DRUGS]
OTHER_AMP_ICD9 = ["8411", "8412", "8413"]  # toe, through foot, ankle disarticulation
OTHER_AMP_ICD10_PREFIXES = ("0Y6M", "0Y6N", "0Y6P", "0Y6Q", "0Y6R", "0Y6S", "0Y6T", "0Y6U",
                            "0Y6V", "0Y6W", "0Y6X", "0Y6Y")  # foot, toes
CLTI_ICD9 = ["44022", "44023", "44024", "7854"]
CLTI_ICD10_PREFIXES = ("I7022", "I7023", "I7024", "I7025", "I7026", "I96")
CLTI_ICD10_REGEX = r"^E(08|09|10|11|13)52"  # diabetes with peripheral angiopathy with gangrene
ATHERO_PAD_ICD9_PREFIX, ATHERO_PAD_ICD10_PREFIX = "4402", "I702"

def mimic(subdir, table):
    return os.path.join(BASE_DIR, subdir, f"{table}.csv.gz")


def data(fname):
    return os.path.join(DATA_DIR, fname)


def results_tables(fname):
    return os.path.join(RES_DIR, "tables", fname)


def results_figures(fname):
    return os.path.join(RES_DIR, "figures", fname)


def log_path(fname):
    return os.path.join(LOG_DIR, fname)
