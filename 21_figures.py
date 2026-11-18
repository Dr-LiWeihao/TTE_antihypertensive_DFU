"""
build_bmc_figures.py (copied from Version_2_JEI/scripts/build_jei_figures.py 2026-11-10; only OUT changed)
---------------------
Main-text figures (Fig1-Fig3) and Online Resource 1 figures (Fig. S1-S5)
for the Journal of Endocrinological Investigation submission, built
directly from the analysis outputs (results/tables/*.csv, data/ccw_*.parquet).
No statistic is typed in by hand.

JEI artwork rules applied (Springer "Artwork and Illustrations Guidelines"):
  - width <= 174 mm, height <= 234 mm, lettering Arial 7-9 pt
  - no titles/captions inside the artwork (captions live in the manuscript)
  - panel labels lowercase bold (a, b, c)
  - files named Fig1, Fig2, ...; TIFF 600 dpi (combination art) + EPS/PDF
    with embedded (Type 42) fonts
  - markers differ by shape/fill, not colour alone (accessibility)

Weights come from scripts/06_primary_analysis.load_weighted() -- the same
function the primary analysis uses -- so the curves match the Cox models.

Output: manuscript/Version_4_BMC/figures/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
from lifelines import KaplanMeierFitter
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from importlib import import_module
primary = import_module("06_primary_analysis")

import config as cfg
RES = os.path.join(cfg.RES_DIR, "tables")
DATA = cfg.DATA_DIR
OUT = os.path.join(cfg.OUT_DIR, "manuscript_figures")
os.makedirs(OUT, exist_ok=True)

MM = 1 / 25.4
FULL_W = 174 * MM
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COMPS = ["CCB", "betablocker", "diuretic"]
COMP_LABEL = {"CCB": "CCB", "betablocker": "Beta-blocker", "diuretic": "Diuretic"}
OUTCOMES = ["afs", "mortality", "amputation"]
OUTCOME_LABEL = {"afs": "Amputation-free survival (primary)",
                 "mortality": "All-cause mortality",
                 "amputation": "Major amputation"}
ACEI_COLOR, COMP_COLOR = "#2166AC", "#B2182B"
KM_MAX_DAYS = 365  # curves shown to 1 year; beyond this very few admissions remain at risk

COV_LABEL = {
    "age_at_admission": "Age", "gender_male": "Male sex", "mi": "Myocardial infarction",
    "chf": "Congestive heart failure", "cerebrovascular": "Cerebrovascular disease",
    "chronic_pulmonary": "Chronic pulmonary disease", "renal": "Renal disease", "sepsis": "Sepsis",
    "charlson_comorbidity_index": "Charlson Comorbidity Index",
    "baseline_creatinine": "Baseline creatinine", "creatinine_unknown": "Creatinine missing",
    "baseline_sbp": "Baseline systolic BP", "baseline_dbp": "Baseline diastolic BP",
    "bp_unknown": "Blood pressure missing", "emergency_admission": "Emergency/urgent admission",
    "icu_stay_flag": "ICU stay", "race_Black": "Race: Black", "race_Hispanic": "Race: Hispanic",
    "race_Other/Unknown": "Race: other/unknown", "race_White": "Race: White",
}

results = pd.read_csv(os.path.join(RES, "primary_results_ALL.csv"))
results["comp"] = results["comparison"].str.replace("ACEI_ARB_vs_", "", regex=False)
BANDS = pd.read_csv(os.path.join(RES, "km_bands.csv"))


def res_row(comp, outcome):
    return results[(results.comp == comp) & (results.outcome == outcome)].iloc[0]


def save(fig, name, raster_only=False, eps=True):
    """TIFF @600 dpi (LZW, RGB) + vector EPS/PDF. Returns list of paths."""
    base = os.path.join(OUT, name)
    fig.savefig(base + ".png", dpi=600)
    Image.open(base + ".png").convert("RGB").save(base + ".tif", dpi=(600, 600), compression="tiff_lzw")
    os.remove(base + ".png")
    if not raster_only:
        if eps:   # EPS cannot hold transparency (KM confidence bands); PDF + TIFF only for those figures
            fig.savefig(base + ".eps")
        elif os.path.exists(base + ".eps"):
            os.remove(base + ".eps")
        fig.savefig(base + ".pdf")
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"{name}: {w/MM:.0f} x {h/MM:.0f} mm")


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Fig 1 -- cohort derivation and clone-censor-weight flow (primary outcome)
# ---------------------------------------------------------------------------
def build_fig1():
    # Cohort counts: logs/01_cohort_extraction.log (546,028 = all MIMIC-IV v3.1 admissions, all adults;
    # 5,787 / 2,511 diabetic-foot admissions / patients). Every other count is read from the data or from
    # scripts/05 build_ccw_dataset (verified identical to the saved CCW datasets). Mortality-analysis flow and
    # "following neither" / in-grace counts added 2026-11-13 (review v3, R3-m3).
    ccw_mod = import_module("05_ccw_clone_censor")
    elig = pd.read_parquet(os.path.join(DATA, "eligible_cohort.parquet"))
    n_adm, n_pat = len(elig), elig.subject_id.nunique()
    n_all, n_dfu_adm, n_dfu_pat = 546028, 5787, 2511
    out = pd.read_parquet(os.path.join(DATA, "outcomes.parquet"))
    n_afs = int(out.loc[out.hadm_id.isin(elig.hadm_id), "afs_eligible"].sum())
    built = {(c, o): ccw_mod.build_ccw_dataset(48, c, o, log=lambda m: None) for c in COMPS for o in ["afs", "mortality"]}
    for c in COMPS:   # the figure must match the saved datasets used by every analysis
        for o in ["afs", "mortality"]:
            assert len(built[(c, o)]) == len(pd.read_parquet(os.path.join(DATA, f"ccw_{c}_{o}.parquet")))

    H = 196
    fig, ax = plt.subplots(figsize=(FULL_W, H * MM))
    ax.set_xlim(0, 174); ax.set_ylim(0, H); ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F4F7", bold=False, fs=7.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec="#33465C", lw=0.7))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
                linespacing=1.3)
        return (x, y, w, h)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     color="#33465C", lw=0.7))

    box(80, 189, 96, 9, f"All admissions in MIMIC-IV v3.1 (adults only)\nn = {n_all:,}")
    box(80, 173, 96, 10, f"Admissions with diabetic foot ulcer or gangrene\nn = {n_dfu_adm:,} admissions ({n_dfu_pat:,} patients)")
    arrow(80, 184.5, 80, 178)
    box(80, 154, 96, 13, f"Eligible: documented hypertension diagnosis\nn = {n_adm:,} admissions ({n_pat:,} patients)\n"
                         f"Time zero = admission (no survival requirement)")
    arrow(80, 168, 80, 160.5)
    box(150, 164, 40, 9, f"Excluded: no hypertension\ndiagnosis (n = {n_dfu_adm - n_adm:,})", fc="white")
    ax.plot([80, 130], [164, 164], color="#33465C", lw=0.7)
    box(80, 134, 120, 13, f"Mortality analysis: all {n_adm:,} admissions\n"
                          f"Amputation-free survival (primary) and major amputation: {n_afs:,} admissions\n"
                          f"(excluding {n_adm - n_afs} with major amputation on the day of admission)")
    arrow(80, 147.5, 80, 140.5)

    xs = [30, 87, 144]
    ax.plot([30, 144], [123, 123], color="#33465C", lw=0.7)
    ax.plot([87, 87], [127.5, 123], color="#33465C", lw=0.7)
    for x, c in zip(xs, COMPS):
        da, dm = built[(c, "afs")], built[(c, "mortality")]
        nf = da.attrs["n_follow"]
        lab = COMP_LABEL[c] if c == "CCB" else COMP_LABEL[c].lower()
        arrow(x, 123, x, 118.5)
        box(x, 115, 54, 7, f"ACEI/ARB vs {lab}", fc="#DDE7F0", bold=True)
        box(x, 96.5, 54, 24, f"Regimen at 48 h*\nACEI/ARB: {nf['ACEI_ARB']:,}\n{COMP_LABEL[c]}, no ACEI/ARB: {nf[c]:,}\n"
                           f"Neither strategy: {nf['neither']:,}\nEvent within 48 h: {da.attrs['n_shared_events']:,}\n"
                           f"({da.attrs['n_dev_before_event']} after an ACEI/ARB order)", fs=7)
        arrow(x, 111.5, x, 108)
        n_clone = 2 * n_afs
        n_cens = n_clone - len(da)
        box(x, 75, 54, 14, f"Cloned into both strategies: {n_clone:,}\nArtificially censored at 48 h\n"
                           f"n = {n_cens:,} ({100 * n_cens / n_clone:.1f}%)", fc="#F7E6E6", fs=7)
        arrow(x, 86, x, 82)
        box(x, 57, 54, 13, f"Retained and weighted*\nACEI/ARB n = {(da.clone_arm == 'ACEI_ARB').sum():,}\n"
                           f"{COMP_LABEL[c]} n = {(da.clone_arm != 'ACEI_ARB').sum():,}", fc="#E4F0E4", fs=7)
        arrow(x, 68, x, 63.5)
        nm_c = 2 * n_adm - len(dm)
        box(x, 33, 54, 25, f"Mortality analysis ({n_adm:,} admissions)\nDeaths within 48 h: {dm.attrs['n_shared_events']:,}\n"
                           f"Artificially censored: {nm_c:,} ({100 * nm_c / (2 * n_adm):.1f}%)\n"
                           f"Retained and weighted\nACEI/ARB n = {(dm.clone_arm == 'ACEI_ARB').sum():,}\n"
                           f"{COMP_LABEL[c]} n = {(dm.clone_arm != 'ACEI_ARB').sum():,}", fc="white", fs=7)
    ax.text(87, 12, "* Amputation-free survival population. Clones with an outcome event within the 48-h grace period are retained in both arms,\n"
                    "unless an ACEI/ARB was ordered before the event (then in the ACEI/ARB arm only).",
            ha="center", va="center", fontsize=6.8, style="italic", linespacing=1.3)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "Fig1")


# ---------------------------------------------------------------------------
# Fig 3 -- forest plot, all outcomes x comparators (48-h grace period)
# ---------------------------------------------------------------------------
def build_fig_forest():
    rows = []
    for o in OUTCOMES:
        rows.append(("header", OUTCOME_LABEL[o], None))
        for c in COMPS:
            rows.append(("data", COMP_LABEL[c], res_row(c, o)))
    fig = plt.figure(figsize=(FULL_W, 110 * MM))
    ax = fig.add_axes([0.53, 0.12, 0.44, 0.84])
    n = len(rows)
    ys = np.arange(n)[::-1]
    ax.set_ylim(-0.8, n - 0.3)
    hdr_x = -1.12  # axes fraction, left text columns
    for y, (kind, label, r) in zip(ys, rows):
        if kind == "header":
            ax.text(hdr_x, y, label, transform=ax.get_yaxis_transform(), fontsize=8, fontweight="bold", va="center")
            continue
        sig = bool(r.passes_bonferroni_within_family)
        ax.errorbar(r.HR, y, xerr=[[r.HR - r.HR_CI_lower], [r.HR_CI_upper - r.HR]], fmt="D" if sig else "o",
                    color="black", mfc="black" if sig else "white", mec="black", ms=4.5, capsize=2, elinewidth=0.8,
                    mew=0.8, zorder=3)
        p = r.cox_p_value
        ptxt = "<0.001" if p < 0.001 else f"{p:.3f}"
        ax.text(hdr_x + 0.04, y, label, transform=ax.get_yaxis_transform(), fontsize=7.5, va="center")
        ax.text(-0.52, y, f"{r.HR:.2f} ({r.HR_CI_lower:.2f}–{r.HR_CI_upper:.2f})", transform=ax.get_yaxis_transform(),
                fontsize=7.5, va="center")
        ax.text(-0.13, y, ptxt, transform=ax.get_yaxis_transform(), fontsize=7.5, va="center")
    top = n - 0.3
    for x, t in [(hdr_x, "Outcome / comparator"), (-0.52, "HR (95% CI)"), (-0.13, "p")]:
        ax.text(x, 1.0, t, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
    ax.axvline(1, color="0.4", ls="--", lw=0.6)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1.0, 2.0]
    lo = min(results.HR_CI_lower.min(), 0.5) * 0.9
    hi = max(results.HR_CI_upper.max(), 1.5) * 1.1
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Hazard ratio, ACEI/ARB vs comparator (log scale)")
    ax.text(0.02, -0.16, "Favours ACEI/ARB", transform=ax.transAxes, fontsize=7, ha="left", va="top")
    ax.text(0.98, -0.16, "Favours comparator", transform=ax.transAxes, fontsize=7, ha="right", va="top")
    save(fig, "Fig3")


# ---------------------------------------------------------------------------
# Weighted KM helper
# ---------------------------------------------------------------------------
def km_panel(ax, comp, outcome, show_ylabel=True, show_risk=False):
    df, _ = primary.load_weighted(comp, outcome)
    for arm, label, color, ls in [(1, "ACEI/ARB", ACEI_COLOR, "-"), (0, COMP_LABEL[comp], COMP_COLOR, "--")]:
        sub = df[df.treated == arm]
        kmf = KaplanMeierFitter().fit(sub["time"], sub["event"], weights=sub["combined_weight"], label=label)
        sf = kmf.survival_function_
        sf = sf[sf.index <= KM_MAX_DAYS]
        # Pointwise 95% bootstrap band (scripts/19_km_bootstrap_bands.py), added 2026-11-13 (review v3, R3-m3)
        bd = BANDS[(BANDS.comparison == comp) & (BANDS.outcome == outcome) &
                   (BANDS.arm == ("ACEI_ARB" if arm == 1 else comp))]
        ax.fill_between(bd["day"], bd["lo"], bd["hi"], step="post", color=color, alpha=0.15, lw=0)
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=color, ls=ls, lw=1.0, label=label)
    r = res_row(comp, outcome)
    txt = f"HR {r.HR:.2f} (95% CI {r.HR_CI_lower:.2f}–{r.HR_CI_upper:.2f})"
    if show_risk:
        f1 = lambda v: f"{100 * v:.1f}".replace("-", "\u2212")
        txt = (f"1-year risk {f1(r.risk_treated_1y)}% vs {f1(r.risk_control_1y)}%\n"
               f"RD {f1(r.rd_1y)} (95% CI {f1(r.rd_1y_CI_lower)} to {f1(r.rd_1y_CI_upper)})\n" + txt)
    ax.text(0.97, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, linespacing=1.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, KM_MAX_DAYS)
    ax.set_xlabel("Days since admission")
    if show_ylabel:
        ax.set_ylabel("Weighted event-free probability")
    ax.legend(loc="center left", frameon=False, handlelength=2.2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # Numbers at risk (unweighted retained clones), added 2026-11-10 (review v2, R3-N1)
    ticks = [0, 90, 180, 270, 365]
    ax.set_xticks(ticks)
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0, -0.31, f"Number at risk (ACEI/ARB; {COMP_LABEL[comp]})", transform=tr, ha="left", va="center",
            fontsize=6, style="italic")
    for k, (arm, color) in enumerate([(1, ACEI_COLOR), (0, COMP_COLOR)]):
        tt = df.loc[df.treated == arm, "time"].values
        for x in ticks:
            ax.text(x, -0.40 - 0.09 * k, f"{int((tt >= x).sum()) if x > 0 else len(tt):,}", transform=tr,
                    ha="left" if x == 0 else ("right" if x == ticks[-1] else "center"), va="center",
                    fontsize=6, color=color)


def build_fig_km():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 80 * MM), sharey=True)
    for i, (ax, c) in enumerate(zip(axes, COMPS)):
        km_panel(ax, c, "afs", show_ylabel=(i == 0), show_risk=True)
        panel_label(ax, "abc"[i], x=-0.18 if i == 0 else -0.08)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.33, top=0.93, wspace=0.16)
    save(fig, "Fig4", eps=False)


# ---------------------------------------------------------------------------
# Fig 2 -- from crude to fully adjusted (09 confounding decomposition, AFS)
# ---------------------------------------------------------------------------
STEP_ORDER = ["M0_crude", "M1_demographics", "M2b_demographics_other", "M2a_demographics_cardiorenal", "M3_full"]
STEP_LABEL = {"M0_crude": "Crude", "M1_demographics": "Demographics",
              "M2b_demographics_other": "Demographics + other factors",
              "M2a_demographics_cardiorenal": "Demographics + cardiorenal",
              "M3_full": "Fully adjusted (primary)"}


def build_fig_stepwise():
    dec = pd.read_csv(os.path.join(RES, "confounding_decomposition.csv"))
    dec["comp"] = dec["comparison"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 58 * MM), sharey=True)
    y = np.arange(len(STEP_ORDER))[::-1]
    lo = min(dec.loc[dec.outcome == "afs", "HR_CI_lower"].min(), 0.5) * 0.92
    hi = max(dec.loc[dec.outcome == "afs", "HR_CI_upper"].max(), 1.5) * 1.08
    for i, (ax, c) in enumerate(zip(axes, COMPS)):
        sub = dec[(dec.outcome == "afs") & (dec.comp == c)].set_index("model").loc[STEP_ORDER]
        for yy, (m, r) in zip(y, sub.iterrows()):
            full = m == "M3_full"
            ax.errorbar(r.HR, yy, xerr=[[r.HR - r.HR_CI_lower], [r.HR_CI_upper - r.HR]], fmt="s" if full else "o",
                        color="black", mfc="black" if full else "white", ms=4, capsize=1.8, elinewidth=0.8, mew=0.8)
        ax.axvline(1, color="0.4", ls="--", lw=0.6)
        ax.set_xscale("log"); ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks(y); ax.set_yticklabels([STEP_LABEL[m] for m in STEP_ORDER])
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"vs {COMP_LABEL[c]}", fontsize=8)
        ax.set_xlabel("Hazard ratio")
        panel_label(ax, "abc"[i], x=-0.06, y=1.06)
        for s_ in ["top", "right", "left"]:
            ax.spines[s_].set_visible(False)
    fig.subplots_adjust(left=0.25, right=0.99, bottom=0.2, top=0.86, wspace=0.1)
    save(fig, "Fig2")


# ---------------------------------------------------------------------------
# Fig 5 -- all sensitivity analyses for the primary outcome (added 2026-11-13, review v3 R1-m2/R3-m6)
# Every estimate is read from the analysis CSVs; nothing is typed in.
# ---------------------------------------------------------------------------
def sensitivity_rows():
    g = pd.read_csv(os.path.join(RES, "sensitivity_grace_period.csv"))
    ew = pd.read_csv(os.path.join(RES, "extreme_weight_sensitivity.csv"))
    ew["comp"] = ew["comparison"].str.replace("ACEI_ARB_vs_", "", regex=False)
    cd = pd.read_csv(os.path.join(RES, "sensitivity_comparator_definition.csv"))
    r2 = pd.read_csv(os.path.join(RES, "sensitivity_review2.csv"))
    r3 = pd.read_csv(os.path.join(RES, "sensitivity_review3.csv"))
    mi = pd.read_csv(os.path.join(RES, "sensitivity_mi_bp.csv"))

    def pick(df, comp_col="comparison", **kw):
        out = {}
        for c in COMPS:
            q = df[(df[comp_col] == c) & (df.outcome == "afs")]
            for k, v in kw.items():
                q = q[q[k] == v]
            if len(q):
                r = q.iloc[0]
                out[c] = (r.HR, r.HR_CI_lower, r.HR_CI_upper)
        return out

    def mi_pick():
        return {c: tuple(mi[(mi.comparison == c) & (mi.outcome == "afs")][["HR_mi", "HR_mi_CI_lower", "HR_mi_CI_upper"]].iloc[0])
                for c in COMPS}

    cdd = {}
    for c, key in [("CCB", "ACEI_ARB_vs_CCB_DHP"), ("diuretic", "ACEI_ARB_vs_diuretic_thiazide")]:
        r = cd[(cd.comparison == key) & (cd.outcome == "afs")].iloc[0]
        cdd[c] = (r.HR, r.HR_CI_lower, r.HR_CI_upper)
    prim = {c: (res_row(c, "afs").HR, res_row(c, "afs").HR_CI_lower, res_row(c, "afs").HR_CI_upper) for c in COMPS}
    return [
        ("Primary analysis", None, prim),
        ("Grace period and follow-up", "header", None),
        ("Grace period 24 h", None, pick(g, grace_hours=24)),
        ("Grace period 72 h", None, pick(g, grace_hours=72)),
        ("Follow-up ended at day 365", None, pick(r3, scenario="C3_admin_censoring_365d")),
        ("Weights", "header", None),
        ("Trimmed at 2.5th/97.5th percentile", None, pick(ew, comp_col="comp", weight_rule="trim_2p5_97p5")),
        ("Extreme weights excluded", None, pick(ew, comp_col="comp", weight_rule="exclude_top1pct")),
        ("Censoring weights only", None, pick(r3, scenario="C6_ipcw_only")),
        ("Weights re-estimated in bootstrap", None, pick(r3, scenario="C5_bootstrap_reestimated")),
        ("Covariates and missing data", "header", None),
        ("Baseline-only covariates", None, pick(r2, scenario="B5_baseline_only_covariates")),
        ("Extended censoring model", None, pick(r2, scenario="B7_extended_censoring_model")),
        ("Admission potassium and ED blood pressure", None, pick(r3, scenario="C1_admission_physiology")),
        ("Early limb procedures in censoring model", None, pick(r3, scenario="C2b_early_proc_in_ipcw")),
        ("Calendar era", None, pick(r2, scenario="B6_calendar_era")),
        ("Multiple imputation of outpatient BP\u2020", None, mi_pick()),
        ("Population and strategies", "header", None),
        ("First admission per patient", None, pick(r2, scenario="B4_first_admission")),
        ("Early limb procedure excluded", None, pick(r3, scenario="C2a_exclude_early_limb_proc")),
        ("Any studied class within 48 h", None, pick(r2, scenario="A1c_any_class_48h")),
        ("Symmetric strategies", None, pick(r3, scenario="C4_symmetric_comparator")),
        ("Narrower comparator definition\u2021", None, cdd),
        ("Major amputation from index stay only", None, pick(r2, scenario="A2b_index_stay_only")),
        ("No baseline contraindication", None, pick(r2, scenario="A1b_no_contraindication")),
    ]


def build_fig5_sensitivity():
    rows = sensitivity_rows()
    n = len(rows)
    fig = plt.figure(figsize=(FULL_W, 150 * MM))
    left, width, gap = 0.335, 0.205, 0.022
    ys = np.arange(n)[::-1]
    allv = [v for _, k, d in rows if d for t in d.values() for v in t[1:]]
    lo, hi = min(0.5, min(allv) * 0.95), max(1.6, max(allv) * 1.05)
    for j, c in enumerate(COMPS):
        ax = fig.add_axes([left + j * (width + gap), 0.08, width, 0.86])
        ax.set_ylim(-0.7, n - 0.4)
        for y, (lab, kind, d) in zip(ys, rows):
            if kind == "header":
                if j == 0:
                    ax.text(-1.58, y, lab, transform=ax.get_yaxis_transform(), fontsize=7.2, fontweight="bold", va="center")
                continue
            if j == 0:
                ax.text(-1.58 if lab == "Primary analysis" else -1.52, y, lab, transform=ax.get_yaxis_transform(), fontsize=7,
                        va="center", fontweight="bold" if lab == "Primary analysis" else "normal")
            if d is None or c not in d:
                ax.text(1, y, "–", ha="center", va="center", fontsize=7, color="0.4")
                continue
            hr, l, h = d[c]
            prim = lab == "Primary analysis"
            ax.errorbar(hr, y, xerr=[[hr - l], [h - hr]], fmt="D" if prim else "o", color="black",
                        mfc="black" if prim else "white", ms=3.6 if prim else 3.2, capsize=1.5, elinewidth=0.7, mew=0.7)
        ax.axvline(1, color="0.4", ls="--", lw=0.6)
        ax.axvline(rows[0][2][c][0], color="0.75", ls=":", lw=0.6)
        ax.set_xscale("log"); ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([])
        for s_ in ["top", "right", "left"]:
            ax.spines[s_].set_visible(False)
        ax.set_title(f"vs {COMP_LABEL[c]}", fontsize=8)
        ax.set_xlabel("Hazard ratio")
        panel_label(ax, "abc"[j], x=-0.04, y=1.0)
    save(fig, "Fig5")


# ---------------------------------------------------------------------------
# Online Resource 1 figures
# ---------------------------------------------------------------------------
def build_figS1_schematic():
    fig, ax = plt.subplots(figsize=(FULL_W, 72 * MM))
    ax.set_xlim(0, 100); ax.set_ylim(-4, 50); ax.axis("off")
    x0, x_evt, gx, x_end = 24, 33, 42, 92  # admission, example event (30 h), end of grace period, end of follow-up
    ax.text(0, 47, "a", fontsize=9, fontweight="bold"); ax.text(4, 47, "Outcome event within the grace period, no ACEI/ARB ordered before it (e.g. death at 30 h)", fontsize=8)
    ax.text(0, 23, "b", fontsize=9, fontweight="bold"); ax.text(4, 23, "ACEI/ARB regimen observed by the end of the grace period", fontsize=8)
    for y in [39, 31]:
        ax.plot([x0, x_evt], [y, y], color="black", lw=1); ax.plot(x_evt, y, "x", color="black", ms=5, mew=1)
    ax.text(0, 39, "ACEI/ARB clone", va="center", fontsize=7.5); ax.text(0, 31, "Comparator clone", va="center", fontsize=7.5)
    ax.text(gx + 2, 35, "Event counted in both arms; censoring weight 1", fontsize=7, va="center")
    ax.plot([x0, x_end], [15, 15], color=ACEI_COLOR, lw=1); ax.plot(x_end, 15, ">", color=ACEI_COLOR, ms=4)
    ax.text(0, 15, "ACEI/ARB clone", va="center", fontsize=7.5)
    ax.text(gx + 2, 17.5, "Follows its strategy: followed to outcome or end of follow-up", fontsize=7)
    ax.plot([x0, gx], [7, 7], color=COMP_COLOR, lw=1); ax.plot(gx, 7, "|", color=COMP_COLOR, ms=8, mew=1.2)
    ax.text(0, 7, "Comparator clone", va="center", fontsize=7.5)
    ax.text(gx + 2, 7, "Deviates: censored at 48 h; retained clones reweighted (IPCW)", fontsize=7, va="center")
    for y0, y1 in [(28, 42), (4, 19)]:
        ax.plot([gx, gx], [y0, y1], color="0.5", ls=":", lw=0.8)
        ax.plot([x0, x0], [y0, y1], color="0.5", ls=":", lw=0.8)
    ax.text(x0, -2, "Admission (time zero)", fontsize=7, ha="center", color="0.3")
    ax.text(gx, -2, "48 h", fontsize=7, ha="center", color="0.3")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    save(fig, "FigS1")


def build_figS2_love():
    fig, axes = plt.subplots(3, 3, figsize=(FULL_W, 210 * MM), sharex=True)
    for j, c in enumerate(COMPS):
        for i, o in enumerate(OUTCOMES):
            ax = axes[i, j]
            b = pd.read_csv(os.path.join(RES, f"balance_table_{c}_{o}.csv"))
            b["label"] = b.covariate.map(COV_LABEL).fillna(b.covariate)
            y = np.arange(len(b))[::-1]
            ax.scatter(b.smd_unweighted.abs(), y, marker="o", facecolors="white", edgecolors="0.4", s=10, lw=0.6,
                       label="Unweighted")
            ax.scatter(b.smd_weighted.abs(), y, marker="s", color="black", s=9, label="Weighted")
            ax.axvline(0.1, color="0.4", ls="--", lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(b.label if j == 0 else [], fontsize=6)
            ax.tick_params(axis="y", length=0)
            if i == 0:
                ax.set_title(f"vs {COMP_LABEL[c]}", fontsize=8)
            if j == 2:
                ax.text(1.03, 0.5, OUTCOME_LABEL[o].replace(" (primary)", ""), transform=ax.transAxes, rotation=270,
                        va="center", fontsize=7.5)
            if i == 2:
                ax.set_xlabel("Absolute SMD")
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.96, bottom=0.05, hspace=0.12, wspace=0.08)
    save(fig, "FigS2")


def build_figS3_km_secondary():
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 160 * MM), sharey=True)
    for i, o in enumerate(["mortality", "amputation"]):
        for j, c in enumerate(COMPS):
            km_panel(axes[i, j], c, o, show_ylabel=(j == 0))
            panel_label(axes[i, j], "abcdef"[3 * i + j], x=-0.2 if j == 0 else -0.08)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.96, wspace=0.16, hspace=0.75)
    save(fig, "FigS4", eps=False)


def build_figS4_weights():
    fig, axes = plt.subplots(3, 3, figsize=(FULL_W, 140 * MM))
    for j, c in enumerate(COMPS):
        df, _ = primary.load_weighted(c, "afs")
        for i, (col, lab) in enumerate([("iptw", "IPTW"), ("ipcw_weight", "IPCW"), ("combined_weight", "Combined weight")]):
            ax = axes[i, j]
            ax.hist(df[col], bins=40, color="0.55", edgecolor="none")
            ax.axvline(df[col].median(), color="black", ls=":", lw=0.8)
            ax.text(0.97, 0.92, f"median {df[col].median():.2f}\n99th pct {df[col].quantile(0.99):.2f}\nmax {df[col].max():.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
            if i == 0:
                ax.set_title(f"vs {COMP_LABEL[c]}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{lab}\nCount")
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.05, hspace=0.35, wspace=0.25)
    save(fig, "FigS3")


def build_figS5_grace():
    sens = pd.read_csv(os.path.join(RES, "sensitivity_grace_period.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 70 * MM), sharey=True)
    marks = {24: "o", 48: "D", 72: "s"}
    for k, (ax, o) in enumerate(zip(axes, OUTCOMES)):
        y = 0; yt = []; yl = []
        for c in COMPS:
            for g in [24, 48, 72]:
                r = sens[(sens.comparison == c) & (sens.outcome == o) & (sens.grace_hours == g)].iloc[0]
                sig = r.cox_p_value < 0.05
                ax.errorbar(r.HR, y, xerr=[[r.HR - r.HR_CI_lower], [r.HR_CI_upper - r.HR]], fmt=marks[g], color="black",
                            mfc="black" if sig else "white", ms=4, capsize=1.5, elinewidth=0.7, mew=0.7)
                yt.append(y); yl.append(f"{COMP_LABEL[c]}, {g} h")
                y -= 1
            y -= 0.6
        ax.axvline(1, color="0.4", ls="--", lw=0.6)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5, 2]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(min(0.5, sens.HR_CI_lower.min() * 0.9), max(1.5, sens.HR_CI_upper.max() * 1.1))
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.5)
        ax.set_title(OUTCOME_LABEL[o].replace(" (primary)", ""), fontsize=7.5)
        ax.set_xlabel("Hazard ratio")
        panel_label(ax, "abc"[k], x=-0.05 if k else -0.55, y=1.07)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.17, top=0.88, wspace=0.12)
    save(fig, "FigS5")


if __name__ == "__main__":
    build_fig1(); build_fig_stepwise(); build_fig_forest(); build_fig_km(); build_fig5_sensitivity()
    build_figS1_schematic(); build_figS2_love(); build_figS3_km_secondary(); build_figS4_weights(); build_figS5_grace()
