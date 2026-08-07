"""Publication figures.

Reads the regenerated numbers from results/ and writes figures/ in four
formats: PNG for review, LZW-compressed TIFF, and EPS plus SVG as editable
vector artwork. Transparency is avoided throughout because EPS does not
support it.

Units and spelling follow the target journal: concentrations in mg l^-1, doses
in mg kg^-1, UK spelling.

Run with:  PYTHONPATH=. python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS = Path("results")
FIGDIR = Path("figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

# Colourblind-safe palette (Wong 2011).
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000", "grey": "#999999",
    "lightgrey": "#DDDDDD",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    "ps.fonttype": 42,
})

CONC = r"Concentration (mg l$^{-1}$)"
DOSE = r"Total dose (mg kg$^{-1}$)"


def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_csv(name):
    p = RESULTS / name
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], {h: [] for h in rows[0]}
    for r in rows[1:]:
        for h, v in zip(header, r):
            data[h].append(v)
    out = {}
    for h, v in data.items():
        try:
            out[h] = np.array([float(x) for x in v])
        except ValueError:
            out[h] = np.array(v, dtype=object)
    return out


def group_by(table, key):
    """Split a long-format table into a dict of sub-tables keyed by a column."""
    if table is None:
        return {}
    idx = defaultdict(list)
    for i, k in enumerate(table[key]):
        idx[k].append(i)
    return {k: {c: table[c][np.array(v)] for c in table} for k, v in idx.items()}


def save(fig, stem):
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300)
    fig.savefig(FIGDIR / f"{stem}.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(FIGDIR / f"{stem}.eps", format="eps")
    fig.savefig(FIGDIR / f"{stem}.svg", format="svg")
    plt.close(fig)
    print(f"  saved {stem} (png, tiff, eps, svg)")


def panel_label(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=9)


# ---------------------------------------------------------------------------
# Figure 1: model schematic
# ---------------------------------------------------------------------------

def fig1_schematic():
    fig, ax = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.0, edgecolor="black", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.1,
                                     color="black"))

    box(0.3, 8.3, 3.4, 1.1, "Layer 0  Dosing regimen\nbolus or infusion",
        CB["lightgrey"])
    box(0.3, 6.4, 3.4, 1.3,
        "Layer 1  Pharmacokinetics\nS- and R-ketamine, norketamine,\n"
        "hydroxynorketamine (calibrated)", CB["sky"])
    box(0.3, 4.5, 3.4, 1.3,
        "Layer 2  Receptor occupancy\npyramidal and interneuron\n"
        "open-channel block", CB["yellow"])

    box(4.5, 5.9, 2.7, 1.5,
        "Layer 3a  Protective\npyramidal block raises the\n"
        "spreading depolarisation\nthreshold", CB["green"])
    box(4.5, 3.2, 2.7, 1.5,
        "Layer 3b  Toxic\ninterneuron block causes\ndisinhibition above a\n"
        "threshold", CB["red"])

    box(8.0, 4.3, 1.8, 1.5, "Layers 4 and 5\nNet injury\nand clinical\nlimits",
        CB["orange"])
    box(4.5, 1.1, 5.3, 1.0,
        "Emergent dose-response window", CB["purple"])

    arrow(2.0, 8.3, 2.0, 7.7)
    arrow(2.0, 6.4, 2.0, 5.8)
    arrow(3.7, 5.3, 4.5, 6.4)
    arrow(3.7, 5.0, 4.5, 3.9)
    arrow(7.2, 6.4, 8.0, 5.5)
    arrow(7.2, 3.8, 8.0, 4.8)
    arrow(8.9, 4.3, 7.0, 2.1)
    ax.text(0.3, 9.6, "Multiscale model of ketamine neuroprotection",
            fontsize=10, fontweight="bold")
    save(fig, "fig1_schematic")


# ---------------------------------------------------------------------------
# Figure 2: pharmacokinetic calibration and external evaluation
# ---------------------------------------------------------------------------

def fig2_pk():
    model = group_by(load_csv("overlay_model.csv"), "panel")
    obs = group_by(load_csv("overlay_observed.csv"), "panel")
    cal = load_json("l1_calibration.json")
    if not model:
        print("  fig2 skipped (no overlay data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), layout="constrained")

    def draw(ax, panels, letter, title, xlim=None):
        for (key, colour, marker, label) in panels:
            if key in model:
                ax.plot(model[key]["t_h"], model[key]["conc_mgL"],
                        color=colour, linewidth=1.3, label=f"{label} (model)")
            if key in obs:
                ax.plot(obs[key]["t_h"], obs[key]["observed_mgL"],
                        linestyle="none", marker=marker, markersize=4,
                        markerfacecolor="none", color=colour,
                        label=f"{label} (observed)")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(CONC)
        if xlim:
            ax.set_xlim(*xlim)
        ax.grid(color="#DDDDDD", linewidth=0.5)
        ax.legend(frameon=False)
        panel_label(ax, letter, title)

    draw(axes[0, 0],
         [("kamp_S_ket_rac", CB["blue"], "o", "S-ketamine"),
          ("kamp_R_ket_rac", CB["red"], "s", "R-ketamine")],
         "A", "Calibration: parent, racemate", xlim=(0, 8))
    draw(axes[0, 1],
         [("kamp_S_nk_rac", CB["green"], "o", "S-norketamine"),
          ("kamp_R_nk_rac", CB["purple"], "s", "R-norketamine")],
         "B", "Calibration: norketamine, racemate", xlim=(0, 8))
    draw(axes[1, 0],
         [("hasan_S_ket", CB["blue"], "o", "S-ketamine"),
          ("hasan_R_ket", CB["red"], "s", "R-ketamine")],
         "C", "Held out: 5 mg intravenous racemate", xlim=(0, 12))

    ax = axes[1, 1]
    roles = {"calibration": (CB["blue"], "o", "Calibration"),
             "external": (CB["orange"], "^", "Held out"),
             "descriptive": (CB["grey"], "x", "Not calibrated")}
    for role, (colour, marker, label) in roles.items():
        xs, ys = [], []
        for key, tab in obs.items():
            if tab["role"][0] != role:
                continue
            xs.extend(tab["observed_mgL"].tolist())
            ys.extend(tab["predicted_mgL"].tolist())
        xs, ys = np.array(xs), np.array(ys)
        keep = (xs > 0) & (ys > 0)
        if keep.sum():
            ax.plot(xs[keep], ys[keep], linestyle="none", marker=marker,
                    markersize=4, markerfacecolor="none", color=colour,
                    label=label)
    lo, hi = 1e-4, 1.0
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9)
    ax.plot([lo, hi], [2 * lo, 2 * hi], color="black", linewidth=0.6,
            linestyle="--")
    ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], color="black", linewidth=0.6,
            linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Observed (mg l$^{-1}$)")
    ax.set_ylabel(r"Predicted (mg l$^{-1}$)")
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    sub = ""
    if cal:
        a = cal["fit_quality"]["after"]
        sub = (f"calibration {a['calibration']['MAPE_percent']:.0f}%, "
               f"held out {a['external']['MAPE_percent']:.0f}% error")
    panel_label(ax, "D", f"Predicted versus observed\n{sub}" if sub
                else "Predicted versus observed")
    save(fig, "fig2_pk_calibration")


# ---------------------------------------------------------------------------
# Figure 3: dose-response window and its mechanistic origin
# ---------------------------------------------------------------------------

def fig3_window():
    w = load_csv("window_doseresponse.csv")
    ws = load_json("window_summary.json")
    em = load_json("emergence.json")
    if w is None:
        print("  fig3 skipped")
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")

    ax = axes[0, 0]
    ax.plot(w["dose_mgkg"], w["I_final"], color=CB["blue"], linewidth=1.5)
    if ws:
        ax.axvspan(ws["window_lower"], ws["window_upper"],
                   color=CB["lightgrey"], zorder=0)
        ax.axvline(ws["optimal_dose"], color=CB["red"], linewidth=1.0,
                   linestyle="--")
        ax.plot([ws["optimal_dose"]], [ws["optimal_injury"]], marker="o",
                color=CB["red"], markersize=5)
    ax.set_xlabel(DOSE)
    ax.set_ylabel("Net injury (arbitrary units)")
    panel_label(ax, "A", "Dose response")
    ax.grid(color="#DDDDDD", linewidth=0.5)
    # The protective limb is shallow enough to be invisible on the full scale,
    # which is itself the point; an inset shows it.
    if ws:
        inset = ax.inset_axes([0.42, 0.46, 0.55, 0.48])
        m = w["dose_mgkg"] <= 0.85
        inset.plot(w["dose_mgkg"][m], w["I_final"][m], color=CB["blue"],
                   linewidth=1.3)
        inset.axvspan(ws["window_lower"], ws["window_upper"],
                      color=CB["lightgrey"], zorder=0)
        inset.axvline(ws["optimal_dose"], color=CB["red"], linewidth=0.9,
                      linestyle="--")
        inset.tick_params(labelsize=6.5)
        inset.set_title("detail", fontsize=6.5, pad=2)
        inset.grid(color="#EEEEEE", linewidth=0.4)

    ax = axes[0, 1]
    ax.plot(w["dose_mgkg"], w["SD_burden"], color=CB["green"], linewidth=1.4,
            label="Spreading depolarisation burden")
    ax.set_xlabel(DOSE)
    ax.set_ylabel("Burden (arbitrary units)", color=CB["green"])
    ax2 = ax.twinx()
    ax2.plot(w["dose_mgkg"], w["T_NRHypo"], color=CB["red"], linewidth=1.4,
             label="Hypofunction injury index")
    ax2.set_ylabel("Injury index (arbitrary units)", color=CB["red"])
    panel_label(ax, "B", "Protective and toxic arms")
    ax.grid(color="#DDDDDD", linewidth=0.5)

    ax = axes[1, 0]
    ax.plot(w["dose_mgkg"], w["B_pyr_peak"], color=CB["blue"], linewidth=1.4,
            label="Pyramidal")
    ax.plot(w["dose_mgkg"], w["B_int_peak"], color=CB["red"], linewidth=1.4,
            label="Interneuron")
    if ws and "B_int_thresh" in ws:
        ax.axhline(ws["B_int_thresh"], color=CB["black"], linewidth=0.8,
                   linestyle=":")
        ax.text(w["dose_mgkg"].max() * 0.55, ws["B_int_thresh"] + 0.02,
                "toxic threshold", fontsize=7)
    ax.set_xlabel(DOSE)
    ax.set_ylabel("Peak occupancy")
    ax.legend(frameon=False)
    panel_label(ax, "C", "Receptor occupancy")
    ax.grid(color="#DDDDDD", linewidth=0.5)

    ax = axes[1, 1]
    if em:
        ax.plot(em["doses"], em["injury_intact"], color=CB["blue"],
                linewidth=1.5, label="Complete model")
        ax.plot(em["doses"], em["injury_ablated"], color=CB["orange"],
                linewidth=1.5, linestyle="--", label="Toxic layer removed")
        ax.legend(frameon=False)
    ax.set_xlabel(DOSE)
    ax.set_ylabel("Net injury (arbitrary units)")
    panel_label(ax, "D", "Emergence test")
    ax.grid(color="#DDDDDD", linewidth=0.5)
    save(fig, "fig3_window")


# ---------------------------------------------------------------------------
# Figure 4: regimen effect and the PODCAST arms
# ---------------------------------------------------------------------------

def fig4_regimen():
    reg = load_json("regimen.json")
    pod = load_json("podcast.json")
    if reg is None:
        print("  fig4 skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")

    ax = axes[0]
    order = ["bolus", "infusion_40min", "infusion_2h", "infusion_4h",
             "infusion_8h"]
    pretty = {"bolus": "bolus", "infusion_40min": "40 min",
              "infusion_2h": "2 h", "infusion_4h": "4 h",
              "infusion_8h": "8 h"}
    names = [k for k in order if k in reg]
    inj = [reg[k]["I_final"] for k in names]
    peak = [reg[k]["C_brain_peak"] for k in names]
    x = np.arange(len(names))
    ax.bar(x, inj, color=CB["blue"], width=0.6, edgecolor="black",
           linewidth=0.5)
    if pod and "placebo" in pod:
        ax.axhline(pod["placebo"]["I_final"], color=CB["black"],
                   linestyle="--", linewidth=0.9)
        ax.text(len(names) - 0.5, pod["placebo"]["I_final"],
                "no drug", fontsize=7, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[n] for n in names], rotation=30, ha="right")
    ax.set_ylabel("Net injury (arbitrary units)")
    ax2 = ax.twinx()
    ax2.plot(x, peak, color=CB["red"], marker="o", markersize=4,
             linewidth=1.2)
    ax2.set_ylabel(r"Peak brain concentration (mg l$^{-1}$)", color=CB["red"])
    ax2.set_yscale("log")
    ax2.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
    ax2.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    ax2.minorticks_off()
    panel_label(ax, "A", r"Matched total dose 0.5 mg kg$^{-1}$")

    ax = axes[1]
    if pod:
        pod_pretty = {"placebo": "placebo",
                      "ket_0.5_bolus": "0.5 mg kg$^{-1}$\nbolus",
                      "ket_1.0_bolus": "1.0 mg kg$^{-1}$\nbolus"}
        names = [k for k in ["placebo", "ket_0.5_bolus", "ket_1.0_bolus"]
                 if k in pod]
        inj = [pod[k]["I_final"] for k in names]
        psych = [pod[k]["psych_max"] for k in names]
        x = np.arange(len(names))
        ax.bar(x, inj, color=CB["blue"], width=0.6, edgecolor="black",
               linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([pod_pretty[n] for n in names], fontsize=7.5)
        ax.set_ylabel("Net injury (arbitrary units)")
        ax2 = ax.twinx()
        ax2.plot(x, psych, color=CB["purple"], marker="s", markersize=4,
                 linewidth=1.2)
        ax2.set_ylabel("Psychotomimetic burden", color=CB["purple"])
    panel_label(ax, "B", "Trial arms reproduced")
    save(fig, "fig4_regimen_podcast")


# ---------------------------------------------------------------------------
# Figure 5: identifiability of the injury coefficients
# ---------------------------------------------------------------------------

def fig5_identifiability():
    idn = load_json("identifiability.json")
    if idn is None:
        print("  fig5 skipped")
        return
    doses = np.array(idn["dose_grid"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), layout="constrained")

    # A: the family of dose-response curves across the toxic weight. Below the
    # critical value the curve falls monotonically and there is no window;
    # above it an interior optimum appears and deepens.
    ax = axes[0]
    sweep = idn["gamma_sweep"]
    crit = idn["critical_values"]["gamma_critical_at_nominal_g_gain"]
    cmap = plt.get_cmap("viridis")
    gammas = np.array([s["gamma"] for s in sweep], dtype=float)
    norm = plt.Normalize(np.log10(gammas.min()), np.log10(gammas.max()))
    for s in sweep:
        curve = s.get("dose_response_normalised")
        if curve is None:
            continue
        ax.plot(doses, curve, color=cmap(norm(np.log10(s["gamma"]))),
                linewidth=1.2,
                linestyle="-" if s["u_shaped"] else "--")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"log$_{10}\ \gamma$", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    ax.set_xlabel(DOSE)
    ax.set_ylabel("Net injury / its minimum")
    ax.set_ylim(0.9, 4.0)
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.text(0.04, 0.95, "dashed: no window", transform=ax.transAxes,
            fontsize=6.5, va="top")
    panel_label(ax, "A", "Window appears above a\ncritical toxic weight")

    # B: optimal dose and window versus the toxic weight.
    ax = axes[1]
    g = [s["gamma"] for s in sweep if s["u_shaped"]]
    opt = [s["optimal_dose"] for s in sweep if s["u_shaped"]]
    upper = [s["window_upper"] for s in sweep if s["u_shaped"]]
    ax.plot(g, opt, color=CB["blue"], marker="o", markersize=4, linewidth=1.3,
            label="optimum")
    ax.plot(g, upper, color=CB["orange"], marker="s", markersize=4,
            linewidth=1.3, label="upper limit")
    ax.axvline(crit, color=CB["red"], linewidth=1.0, linestyle="--")
    ax.text(crit * 1.15, max(upper) * 0.95, "window\ndisappears", fontsize=7,
            color=CB["red"], va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"Hypofunction injury weight $\gamma$")
    ax.set_ylabel(r"Dose (mg kg$^{-1}$)")
    ax.legend(frameon=False)
    ax.grid(color="#DDDDDD", linewidth=0.5)
    panel_label(ax, "B", "Trough moves with toxic weight")

    # C: where the nominal value sits relative to the critical boundary.
    ax = axes[2]
    cm = idn["critical_gamma_map"]
    flat = np.array([v for row in cm["Gamma_over_alpha_critical"]
                     for sub in row for v in sub if v is not None],
                    dtype=float)
    ax.hist(flat, bins=30, color=CB["sky"], edgecolor="black", linewidth=0.4)
    nom = idn["nominal"]["Gamma_over_alpha"]
    ax.axvline(nom, color=CB["red"], linewidth=1.2)
    ax.text(nom * 0.92, ax.get_ylim()[1] * 0.9, "nominal", fontsize=7,
            color=CB["red"], ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Critical toxic weight Γ/α")
    ax.set_ylabel("Coefficient sets")
    panel_label(ax, "C", "Critical versus nominal\ntoxic weight")
    save(fig, "fig5_identifiability")


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------

def figS1_vpop_cyp():
    vp = load_json("vpop_summary.json")
    cy = load_json("cyp2b6.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), layout="constrained")

    ax = axes[0]
    if vp and "key_doses" in vp:
        d = np.array(vp["key_doses"])
        ax.plot(d, vp["median"], color=CB["blue"], linewidth=1.5,
                marker="o", markersize=3.5)
        ax.fill_between(d, vp["p5"], vp["p95"], color=CB["sky"], zorder=0)
    ax.set_xlabel(DOSE)
    ax.set_ylabel("Net injury (arbitrary units)")
    panel_label(ax, "A", "Virtual population")
    ax.grid(color="#DDDDDD", linewidth=0.5)

    ax = axes[1]
    if vp and "optimal_dose_distribution" in vp:
        ax.hist(vp["optimal_dose_distribution"], bins=12, color=CB["sky"],
                edgecolor="black", linewidth=0.4)
    ax.set_xlabel(r"Individual optimal dose (mg kg$^{-1}$)")
    ax.set_ylabel("Subjects")
    panel_label(ax, "B", "Optimal dose distribution")

    ax = axes[2]
    if cy:
        genos = [g for g in ("*1/*1", "*1/*6", "*6/*6") if g in cy]
        for geno, colour in zip(genos, [CB["blue"], CB["orange"], CB["red"]]):
            ax.plot(cy[geno]["doses"], cy[geno]["I_final"], color=colour,
                    linewidth=1.3, label=f"CYP2B6 {geno}")
        ax.legend(frameon=False)
    ax.set_xlabel(DOSE)
    ax.set_ylabel("Net injury (arbitrary units)")
    panel_label(ax, "C", "CYP2B6 genotype")
    ax.grid(color="#DDDDDD", linewidth=0.5)
    save(fig, "figS1_vpop_cyp2b6")


def figS2_sobol():
    s = load_json("sobol.json")
    if s is None:
        print("  figS2 skipped")
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4), layout="constrained")

    labels = s.get("param_labels", s["param_names"])
    st = np.array(s["ST"])
    s1 = np.array(s["S1"])
    order = np.argsort(st)
    y = np.arange(len(order))

    ax = axes[0]
    ax.barh(y - 0.2, st[order], height=0.38, color=CB["blue"],
            xerr=np.array(s["ST_conf"])[order], edgecolor="black",
            linewidth=0.4, error_kw=dict(lw=0.6), label="total order")
    ax.barh(y + 0.2, s1[order], height=0.38, color=CB["orange"],
            xerr=np.array(s["S1_conf"])[order], edgecolor="black",
            linewidth=0.4, error_kw=dict(lw=0.6), label="first order")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=6.5)
    ax.set_xlabel("Sobol index")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "A", "Sensitivity indices")

    ax = axes[1]
    conv = s.get("convergence", [])
    if conv:
        ns = [c["n_base"] for c in conv]
        top = np.argsort(st)[::-1][:5]
        for k, colour in zip(top, [CB["blue"], CB["orange"], CB["green"],
                                   CB["red"], CB["purple"]]):
            ax.plot(ns, [c["ST"][k] for c in conv], marker="o", markersize=3.5,
                    linewidth=1.2, color=colour, label=labels[k])
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.legend(frameon=False, fontsize=6)
    ax.set_xlabel("Base sample size")
    ax.set_ylabel("Total-order index")
    ax.grid(color="#DDDDDD", linewidth=0.5)
    panel_label(ax, "B", "Convergence")

    ax = axes[2]
    pairs = s.get("second_order_run", {}).get("S2_pairs", [])[:8]
    if pairs:
        vals = [abs(p["S2"]) for p in pairs][::-1]
        names = [f"{p['a']} x {p['b']}" for p in pairs][::-1]
        yy = np.arange(len(vals))
        ax.barh(yy, vals, color=CB["green"], edgecolor="black", linewidth=0.4)
        ax.set_yticks(yy)
        ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("|Second-order index|")
    panel_label(ax, "C", "Strongest interactions")
    save(fig, "figS2_sobol")


def main():
    print("Building figures ...")
    fig1_schematic()
    fig2_pk()
    fig3_window()
    fig4_regimen()
    fig5_identifiability()
    figS1_vpop_cyp()
    figS2_sobol()
    print(f"Done. Figures in {FIGDIR}")


if __name__ == "__main__":
    main()
