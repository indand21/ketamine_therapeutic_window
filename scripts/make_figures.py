"""Generate publication-quality figures for the BJA manuscript.

Reads regenerated numbers from results/ and writes PNG (300 dpi)
and TIFF to figures/. Colorblind-safe palette; no em/en dashes in
any label text.

Run with:  PYTHONPATH=. python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS = Path("results")
FIGDIR = Path("figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe palette (Wong 2011).
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000", "grey": "#999999",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})


def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_csv(name):
    p = RESULTS / name
    if not p.exists():
        return None
    import csv
    with p.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = {h: [] for h in header}
    for r in rows[1:]:
        for h, v in zip(header, r):
            try:
                data[h].append(float(v))
            except ValueError:
                data[h].append(np.nan)
    return {h: np.array(v) for h, v in data.items()}


def save(fig, stem):
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300)
    fig.savefig(FIGDIR / f"{stem}.tiff", dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  saved {stem}.png / .tiff")


# ---------------------------------------------------------------------------
# Figure 1: multiscale model schematic
# ---------------------------------------------------------------------------

def fig1_schematic():
    fig, ax = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, color):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=1.0, edgecolor="black", facecolor=color, alpha=0.85)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2, color="black"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=11, linewidth=1.1, color=color))

    box(0.3, 8.4, 3.2, 1.1, "L0 Dosing regimen\n(bolus / infusion)", CB["grey"])
    box(0.3, 6.6, 3.2, 1.2, "L1 PBPK/PK\nS/R-ketamine, NK, HNK\n4 compartments", CB["sky"])
    box(0.3, 4.8, 3.2, 1.2, "L2 NMDAR occupancy\nB(pyramidal), B(interneuron)", CB["yellow"])

    box(4.4, 6.0, 2.6, 1.4, "L3a SD susceptibility\n(protective)\nB(pyr) raises threshold", CB["green"])
    box(4.4, 3.4, 2.6, 1.4, "L3b NRHypo injury\n(toxic)\nB(int) disinhibition", CB["red"])

    box(7.9, 4.5, 1.9, 1.4, "L4/L5\nNet injury +\nclinical limits", CB["orange"])
    box(4.4, 1.2, 5.4, 1.0, "Predicted therapeutic window (U-shaped dose response)", CB["purple"])

    arrow(1.9, 8.4, 1.9, 7.8)
    arrow(1.9, 6.6, 1.9, 6.0)
    arrow(3.5, 5.4, 4.4, 6.4)   # L2 -> L3a
    arrow(3.5, 5.2, 4.4, 4.0)   # L2 -> L3b
    arrow(7.0, 6.5, 7.9, 5.6)   # L3a -> L4
    arrow(7.0, 4.0, 7.9, 5.0)   # L3b -> L4
    arrow(8.8, 4.5, 6.8, 2.2)   # L4 -> window
    ax.text(5.0, 9.4, "Multiscale QSP framework for ketamine neuroprotection",
            fontsize=10, fontweight="bold")
    save(fig, "fig1_schematic")


# ---------------------------------------------------------------------------
# Figure 2: L1 PK validation
# ---------------------------------------------------------------------------

def fig2_pk():
    val = load_json("l1_validation.json")
    hasan_m = load_csv("overlay_hasan_model.csv")
    hasan_s = load_csv("overlay_hasan_digitized_S.csv")
    hasan_r = load_csv("overlay_hasan_digitized_R.csv")
    kamp_m = load_csv("overlay_kamp_model.csv")
    kamp_d = load_csv("overlay_kamp_digitized_S.csv")
    zhao = load_csv("pk_zhao_profile.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), layout="constrained")

    ax = axes[0, 0]
    if hasan_m is not None:
        ax.plot(hasan_m["t_h"], hasan_m["KET_S"], color=CB["blue"], label="S-ketamine (model)")
        ax.plot(hasan_m["t_h"], hasan_m["KET_R"], color=CB["red"], label="R-ketamine (model)")
    if hasan_s is not None:
        ax.scatter(hasan_s["t_h"], hasan_s["conc_mgL"], s=14, color=CB["blue"],
                   marker="o", label="S (Hasan 2021)")
    if hasan_r is not None:
        ax.scatter(hasan_r["t_h"], hasan_r["conc_mgL"], s=14, color=CB["red"],
                   marker="s", label="R (Hasan 2021)")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Plasma conc. (mg/L)")
    ax.set_title("A  IV 5 mg (Hasan 2021)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if kamp_m is not None:
        ax.plot(kamp_m["t_h"], kamp_m["KET_S"], color=CB["blue"], label="S-ketamine (model)")
    if kamp_d is not None:
        ax.scatter(kamp_d["t_h"], kamp_d["conc_mgL"], s=14, color=CB["blue"],
                   marker="o", label="S (Kamp 2020)")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Plasma conc. (mg/L)")
    ax.set_title("B  Escalating infusion (Kamp 2020)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if zhao is not None:
        ax.plot(zhao["t_h"], zhao["KET_S"], color=CB["blue"], label="S-ketamine")
        ax.plot(zhao["t_h"], zhao["KET_R"], color=CB["red"], label="R-ketamine")
        ax.plot(zhao["t_h"], zhao["NK_S"] + zhao["NK_R"], color=CB["green"], label="Norketamine")
        ax.plot(zhao["t_h"], zhao["HNK"], color=CB["orange"], label="HNK")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Plasma conc. (mg/L)")
    ax.set_title("C  0.5 mg/kg IV (40 min)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1)

    ax = axes[1, 1]; ax.axis("off")
    if val is not None:
        ax.set_title(f"D  Structural validation ({val['n_pass']}/{val['n_total']})")
        # check_pk_metrics returns a fixed order; map by metric prefix robustly.
        def short_label(metric):
            if metric.startswith("t"):
                return "t1/2 (h)"
            if metric.startswith("CL_total"):
                return "CL (L/h)"
            if metric.startswith("Vdss"):
                return "Vdss (L)"
            if metric.startswith("HNK"):
                return "HNK:KET"
            if metric.startswith("CL ratio"):
                return "CL S:R"
            return metric[:12]
        cell_text = []
        for c in val["checks"]:
            short = short_label(c["metric"])
            ref = "" if c["ref_low"] is None else f"{c['ref_low']:g}-{c['ref_high']:g}"
            val_s = "" if c["value"] is None else f"{c['value']:.3g}"
            cell_text.append([short, val_s, ref, "pass" if c["passed"] else "fail"])
        tbl = ax.table(cellText=cell_text,
                       colLabels=["Metric", "Model", "Reference", "Result"],
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1.0, 1.3)
    save(fig, "fig2_pk_validation")


# ---------------------------------------------------------------------------
# Figure 3: therapeutic window
# ---------------------------------------------------------------------------

def fig3_window():
    w = load_csv("window_doseresponse.csv")
    ws = load_json("window_summary.json")
    em = load_json("emergence.json")
    if w is None:
        return
    d = w["dose_mgkg"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), layout="constrained")

    ax = axes[0, 0]
    ax.plot(d, w["I_final"], "-o", color=CB["black"], ms=3)
    if ws is not None:
        ax.axvline(ws["optimal_dose"], color=CB["green"], ls="--",
                   label=f"Optimal {ws['optimal_dose']:.2f} mg/kg")
        ax.axvspan(ws["window_lower"], ws["window_upper"], color=CB["green"], alpha=0.12)
    ax.set_xlabel("Dose (mg/kg)"); ax.set_ylabel("Net injury (arb. units)")
    ax.set_title("A  Dose-response window"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(d, w["SD_burden"], "-^", color=CB["green"], ms=3, label="SD burden (protective)")
    axt = ax.twinx()
    axt.plot(d, w["T_NRHypo"], "-v", color=CB["red"], ms=3, label="NRHypo (toxic)")
    ax.set_xlabel("Dose (mg/kg)"); ax.set_ylabel("SD burden", color=CB["green"])
    axt.set_ylabel("NRHypo index", color=CB["red"])
    ax.set_title("B  Protective vs toxic arm"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(d, w["B_pyr_peak"], "-o", color=CB["blue"], ms=3, label="B (pyramidal)")
    ax.plot(d, w["B_int_peak"], "-s", color=CB["red"], ms=3, label="B (interneuron)")
    if ws is not None and "B_int_thresh" in ws:
        ax.axhline(ws["B_int_thresh"], color=CB["grey"], ls=":", label="NRHypo threshold")
    ax.set_xlabel("Dose (mg/kg)"); ax.set_ylabel("Peak occupancy")
    ax.set_title("C  Population segregation"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if em is not None:
        ax.plot(em["doses"], em["injury_intact"], "-o", color=CB["black"], ms=3,
                label="Intact model")
        ax.plot(em["doses"], em["injury_ablated"], "--", color=CB["purple"],
                label="Toxic arm ablated")
    ax.set_xlabel("Dose (mg/kg)"); ax.set_ylabel("Net injury (arb. units)")
    ax.set_title("D  Emergence test"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, "fig3_window")


# ---------------------------------------------------------------------------
# Figure 4: regimen + PODCAST
# ---------------------------------------------------------------------------

def fig4_regimen_podcast():
    reg = load_json("regimen.json")
    pod = load_json("podcast.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")

    ax = axes[0]
    if reg is not None:
        names = [k for k in reg if reg[k] is not None]
        labels = {"bolus": "Bolus", "infusion_40min": "40 min",
                  "infusion_2h": "2 h", "infusion_4h": "4 h", "infusion_8h": "8 h"}
        peak = [reg[n]["C_brain_peak"] for n in names]
        inj = [reg[n]["I_final"] for n in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, inj, 0.4, color=CB["orange"], label="Net injury")
        axt = ax.twinx()
        axt.bar(x + 0.2, peak, 0.4, color=CB["blue"], label="Peak brain conc.")
        ax.set_xticks(x); ax.set_xticklabels([labels.get(n, n) for n in names], rotation=0)
        ax.set_ylabel("Net injury (arb.)", color=CB["orange"])
        axt.set_ylabel("Peak brain conc. (mg/L)", color=CB["blue"])
        ax.set_title("A  Regimen (0.5 mg/kg matched)")
        # no-drug placebo net injury baseline: regimens below it are net protective
        if pod is not None and pod.get("placebo") is not None:
            pl = pod["placebo"]["I_final"]
            ax.axhline(pl, ls="--", lw=1.0, color=CB["grey"], zorder=5)
            ax.text(x[2], pl + 0.6, f"placebo {pl:.0f}", va="bottom", ha="center",
                    fontsize=7, color=CB["grey"], zorder=6)

    ax = axes[1]
    if pod is not None:
        names = [k for k in pod if pod[k] is not None]
        labels = {"placebo": "Placebo", "ket_0.5_bolus": "0.5 mg/kg",
                  "ket_1.0_bolus": "1.0 mg/kg"}
        inj = [pod[n]["I_final"] for n in names]
        psych = [pod[n]["psych_max"] for n in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, inj, 0.4, color=CB["orange"], label="Net injury")
        axt = ax.twinx()
        axt.bar(x + 0.2, psych, 0.4, color=CB["purple"], label="Psych burden")
        ax.set_xticks(x); ax.set_xticklabels([labels.get(n, n) for n in names])
        ax.set_ylabel("Net injury (arb.)", color=CB["orange"])
        axt.set_ylabel("Psychotomimetic burden", color=CB["purple"])
        ax.set_title("B  PODCAST bolus arms")
    save(fig, "fig4_regimen_podcast")


# ---------------------------------------------------------------------------
# Figure 5: virtual population + CYP2B6
# ---------------------------------------------------------------------------

def fig5_vpop_cyp():
    vp = load_json("vpop_summary.json")
    inj = load_csv("vpop_injuries.csv")
    cyp = load_json("cyp2b6.json")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), layout="constrained")

    ax = axes[0]
    if vp is not None:
        kd = np.array(vp["key_doses"])
        ax.plot(kd, vp["median"], "-o", color=CB["red"], ms=4, label="Median")
        ax.fill_between(kd, vp["p5"], vp["p95"], color=CB["red"], alpha=0.15,
                        label="90% interval")
        ax.set_xlabel("Dose (mg/kg)"); ax.set_ylabel("Net injury (arb.)")
        ax.set_title(f"A  Virtual population (N={vp['n_subjects']})")
        ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    if vp is not None:
        od = np.array(vp["optimal_dose_distribution"])
        bins = np.arange(0, 2.26, 0.25) - 0.125
        ax.hist(od, bins=bins, color=CB["sky"], edgecolor="black")
        ax.axvline(vp["population_optimal_dose"], color=CB["red"], ls="--",
                   label=f"Median {vp['population_optimal_dose']:.2f}")
        ax.set_xlabel("Individual optimal dose (mg/kg)"); ax.set_ylabel("Count")
        ax.set_title("B  Optimal dose distribution"); ax.legend()

    ax = axes[2]
    if cyp is not None:
        colors = {"*1/*1": CB["blue"], "*1/*6": CB["orange"], "*6/*6": CB["red"]}
        for geno in ("*1/*1", "*1/*6", "*6/*6"):
            if geno in cyp:
                g = cyp[geno]
                ax.plot(g["doses"], g["I_final"], "-", color=colors[geno],
                        label=f"{geno} (opt {g['optimal_dose']:.2f})")
        ax.set_xlabel("Dose (mg/kg)"); ax.set_ylabel("Net injury (arb.)")
        ax.set_title("C  CYP2B6 genotype"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, "fig5_vpop_cyp2b6")


# ---------------------------------------------------------------------------
# Figure 6: Sobol sensitivity
# ---------------------------------------------------------------------------

def fig6_sobol():
    s = load_json("sobol.json")
    if s is None:
        print("  (no sobol.json; skipping fig6)")
        return
    pretty = {
        "k_off_int": "k_off (interneuron)", "CL_out_S": "CL_out (BBB efflux)",
        "B_int_thresh": "B_int threshold", "CL_in_S": "CL_in (BBB influx)",
        "g_gain": "NRHypo gain", "k_on_S_int": "k_on (interneuron)",
        "Q_per": "Q_per", "k_off_pyr": "k_off (pyramidal)",
        "k_on_S_pyr": "k_on (pyramidal)", "CL_out_HNK": "CL_out (HNK)",
    }
    names = [pretty.get(n, n) for n in s["param_names"]]
    S1 = np.array(s["S1"]); ST = np.array(s["ST"])
    S1c = np.array(s.get("S1_conf", [0] * len(S1)))
    STc = np.array(s.get("ST_conf", [0] * len(ST)))
    order = np.argsort(ST)
    names = [names[i] for i in order]
    S1 = S1[order]; ST = ST[order]; S1c = S1c[order]; STc = STc[order]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    ax.barh(y - 0.2, ST, 0.4, color=CB["blue"], label="Total order (ST)",
            xerr=STc, error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.barh(y + 0.2, S1, 0.4, color=CB["orange"], label="First order (S1)",
            xerr=S1c, error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Sobol index (95% bootstrap CI)")
    n_s = s.get("n_samples", "")
    ax.set_title(f"Global sensitivity of net injury (N={n_s})")
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    save(fig, "fig6_sobol")


def main():
    print("Generating figures...")
    fig1_schematic()
    fig2_pk()
    fig3_window()
    fig4_regimen_podcast()
    fig5_vpop_cyp()
    fig6_sobol()
    print("Done.")


if __name__ == "__main__":
    main()
