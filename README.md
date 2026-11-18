# Antihypertensive drug class and amputation-free survival in diabetic foot disease: target trial emulation (MIMIC-IV)

Analysis code for:

> *Antihypertensive drug class and amputation-free survival in patients hospitalized with diabetic foot disease and hypertension: a target trial emulation.* Submitted to BMC Cardiovascular Disorders.

The study emulates a target trial in hospital admissions for diabetic foot ulcer or gangrene with a hypertension diagnosis in MIMIC-IV v3.1. It compares a regimen containing an ACE inhibitor or angiotensin-receptor blocker (ACEI/ARB) with a regimen containing a calcium-channel blocker (CCB), a beta-blocker or a diuretic and no ACEI/ARB. The design is clone-censor-weight with a 48-h grace period from admission. The primary outcome is amputation-free survival; the secondary outcomes are all-cause mortality and major amputation.

This repository contains only the code that produces the results, tables and figures. It contains no data and no code for writing the manuscript text.

## Data

No data are included. Both databases are available to credentialed users under the PhysioNet Credentialed Health Data License:

- MIMIC-IV v3.1, `hosp/` and `icu/` modules: <https://physionet.org/content/mimiciv/3.1/>
- MIMIC-IV-ED v2.2, `ed/` module (used only by script 16, for emergency department triage blood pressure and medication reconciliation): <https://physionet.org/content/mimic-iv-ed/2.2/>

The scripts read the gzip-compressed CSV files as distributed and never modify them.

Everything the scripts write goes to `output/` (or `OUTPUT_DIR`). That folder holds patient-level derived data. It is git-ignored and must not be shared or uploaded.

## Setup and running

```bash
pip install -r requirements.txt
export MIMIC_IV_DIR=/path/to/mimic-iv-3.1        # folder containing hosp/ and icu/
export MIMIC_IV_ED_DIR=/path/to/mimic-iv-ed-2.2/ed
python run_all.py                                # or: python run_all.py --from 12
```

On Windows PowerShell, use `$env:MIMIC_IV_DIR = "D:\path\to\mimic-iv-3.1"` and the same for `MIMIC_IV_ED_DIR`. Set `OUTPUT_DIR` to write outputs somewhere other than `./output`.

A full run took about 5.5 h on a desktop PC; most of it is the 1,000-replicate patient-level bootstraps. All randomness is seeded (`SEED = 42` in `config.py`; bootstrap: 1,000 patient-level replicates). The figures use the Arial font. The versions in `requirements.txt` are the ones used for the published results.

## Pipeline

| Script | Purpose | Main output (under `output/`) |
|---|---|---|
| `00a_dfu_cohort.py` | Admissions with diabetic foot ulcer or gangrene (ICD-9/ICD-10 definitions in the docstring) | `data/diabetic_foot_cohort_admissions.parquet` |
| `00b_comorbidity_cci.py` | Comorbidities and Charlson index (Quan 2005), sepsis | `data/comorbidity_cci.parquet` |
| `01_cohort_extraction.py` | Eligibility: hypertension diagnosis; all qualifying admissions; time zero = admission | `data/eligible_cohort.parquet` |
| `02_treatment_assignment.py` | Antihypertensive classes ordered within 24/48/72 h, and time of first order | `data/drug_exposure.parquet` |
| `03_covariates.py` | Baseline covariates (demographics, comorbidities, outpatient blood pressure, creatinine, admission type, ICU stay) | `data/covariates.parquet` |
| `04_outcomes.py` | Amputation-free survival, mortality and major amputation (all MIMIC-IV admissions within follow-up) | `data/outcomes.parquet` |
| `05_ccw_clone_censor.py` | Cloning, artificial censoring at the end of the grace period, per-arm IPCW | `data/ccw_<comparator>_<outcome>.parquet` |
| `06_primary_analysis.py` | IPTW × IPCW-weighted Cox models, balance, bootstrap CIs, 1-year risks, proportional-hazards checks | `results/tables/primary_results_ALL.csv`, `balance_table_*.csv` |
| `07_sensitivity_grace_period.py` | 24-h and 72-h grace periods | `results/tables/sensitivity_grace_period.csv` |
| `08_extreme_weight_sensitivity.py` | Alternative handling of extreme weights | `results/tables/extreme_weight_sensitivity.csv` |
| `09_confounding_decomposition.py` | Crude-to-adjusted decomposition with nested covariate sets | `results/tables/confounding_decomposition.csv` |
| `10_sensitivity_comparator_definition.py` | Dihydropyridine CCB and thiazide-only diuretic comparators | `results/tables/sensitivity_comparator_definition.csv` |
| `11_extract_early_labs_icu.py` | Early laboratory values and ICU timing for sensitivity analyses | `data/sensitivity_covariates.parquet` |
| `12_sensitivity_review2.py` | Sensitivity scenarios (no baseline contraindication, index stay only, first admission, baseline-only covariates, calendar era, extended censoring model, any class), descriptives | `results/tables/sensitivity_review2.csv`, `sens_review2_*.csv` |
| `13_amputation_cuminc.py` | Weighted Aalen–Johansen cumulative incidence of major amputation | `results/tables/amputation_cuminc.csv` |
| `14_extract_pad_bbagent_anyamp.py` | Limb ischaemia or gangrene, atherosclerotic PAD, beta-blocker type, any amputation | `data/pad_bb_anyamp.parquet` |
| `15_betablocker_pad_exploration.py` | Comparisons by limb ischaemia and PAD, beta-blocker type, any amputation | `results/tables/bb_pad_exploration.csv` |
| `16_extract_review3.py` | ED triage blood pressure and medication reconciliation, first-order times, persistence, early limb procedures | `data/review3_covariates.parquet` |
| `17_sensitivity_review3.py` | Further sensitivity scenarios (admission physiology, early limb procedures, 365-day follow-up, symmetric comparator, re-estimated weights in the bootstrap, censoring weights only), positivity and ascertainment descriptives | `results/tables/sensitivity_review3.csv`, `review3_*.csv` |
| `18_sensitivity_mi_bp.py` | Multiple imputation of outpatient blood pressure (20 imputations, Rubin's rules) | `results/tables/sensitivity_mi_bp.csv` |
| `19_km_bootstrap_bands.py` | Pointwise bootstrap confidence bands for the weighted Kaplan–Meier curves | `results/tables/km_bands.csv` |
| `20_tables.py` | Tables 2–3 and Additional file 1 Tables S1–S13 and S15–S18 | `manuscript_tables/*.csv` |
| `21_figures.py` | Figs. 1–5 and Additional file 1 Figs. S1–S5 | `manuscript_figures/` |

`config.py` holds the paths, analysis parameters and drug and diagnosis code lists. Table 1 (the target trial protocol) and Table S14 (timing of each analysis) are text tables and are not produced by code.

Scenario scripts take the scenario names as arguments, then `combine` (see `run_all.py`). `17_sensitivity_review3.py check` confirms that the clone-censor-weight function with all optional arguments at their defaults reproduces the primary datasets exactly.

## Notes on the methods

- **Multiple admissions per patient.** All qualifying admissions are included. Cox standard errors are clustered on patient, and bootstrap replicates resample patients together with all of their admissions.
- **Strategies.** The ACEI/ARB strategy is an ACEI or ARB ordered within the 48-h grace period, with other classes allowed. The comparator strategy is the comparator class ordered with no ACEI/ARB. A comparator clone whose first ACEI/ARB order comes before an outcome event inside the grace period is censored as a deviator. Drug classes are identified by generic-name matching on `prescriptions.drug` (lists in `config.py`).
- **Weights.** The censoring model is fitted per arm on clones still at risk at the end of the grace period. Clones with an event inside the grace period get weight 1. The propensity model is fitted with the censoring weights as sample weights, and the product of the two weights is trimmed at the 1st and 99th percentiles.
- **Identifiers.** MIMIC identifiers are read as int32 or int64, or as float64 or nullable Int64 where missing values occur, never as float32, which cannot represent them exactly.

## About the code comments

The scripts are released as they were run for the published results; only the file-path handling was changed so that they run outside the authors' computer. Their comments still refer to the authors' internal design notes (`notes/…`, `CLAUDE.md`), project numbers and review rounds, which are not part of this repository. Scripts `00a` and `00b` were originally part of a companion project and are included so that the pipeline runs from the raw data.

## Citation

If you use this code, please cite the article above (reference details will be added on publication).

## License

MIT (see `LICENSE`). The license covers the code only, not MIMIC-IV data.
