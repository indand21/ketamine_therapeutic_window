"""Quantitative goodness of fit of the L1 pharmacokinetic layer.

The earlier version of this work presented the pharmacokinetic layer as two
visual overlays plus five summary targets, which reviewers judged insufficient
to establish that the pharmacokinetics are adequately reproduced. This script
adds a quantitative prediction-error analysis over every digitized
concentration-time curve available for the model species, covering both
enantiomers of the parent, both enantiomers of norketamine, and total
hydroxynorketamine, in two independent studies and three dosing regimens.

Because no pharmacokinetic parameter was estimated from these curves - the
parameters come from published population analyses - these are prediction
errors, not fit residuals.

Metrics per curve (computed on the digitized points):
  MPE   median prediction error, (pred - obs)/obs, as a percentage (bias)
  MAPE  median absolute prediction error, as a percentage (precision)
  RMSLE root mean squared error of log10 residuals
  F2    fraction of points predicted within two-fold of the observation
  plus observed and predicted Cmax and Tmax.

Run with:  PYTHONPATH=. python scripts/run_pk_goodness_of_fit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.l1_pk import (L1Model, DosingRegimen, DoseEvent,
                       InfusionSegment)
from src.l1_pk.config import default_parameters
from src.validation.l1_pk_validation import (
    kamp_2020_escalating_regimen, REF_WEIGHT_KG,
)
from src.ingestion.digitized_data import (
    HASAN_2021_S_KET_PLASMA, HASAN_2021_R_KET_PLASMA,
    KAMP_2020_S_KET_ESKETAMINE, KAMP_2020_S_KET_RACEMIC,
    KAMP_2020_R_KET_RACEMIC, KAMP_2020_S_NK_ESKETAMINE,
    KAMP_2020_S_NK_RACEMIC, KAMP_2020_R_NK_RACEMIC,
    KAMP_2020_HNK_RACEMIC,
)

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

# Hasan 2021 Fig 1: 5 mg racemic ketamine as an intravenous reference dose.
# 5 mg racemate given intravenously over 30 min (Hasan 2021), not a bolus.
HASAN_REGIMEN = DosingRegimen(
    infusions=[InfusionSegment(start=0.0, end=0.5, rate=5.0 / 0.5)],
    s_fraction=0.5)

CURVES = [
    # (label, curve, species, regimen, sampling site, study)
    ("S-ketamine, 5 mg i.v.", HASAN_2021_S_KET_PLASMA, "KET_S",
     HASAN_REGIMEN, "venous", "Hasan 2021"),
    ("R-ketamine, 5 mg i.v.", HASAN_2021_R_KET_PLASMA, "KET_R",
     HASAN_REGIMEN, "venous", "Hasan 2021"),
    ("S-ketamine, escalating esketamine", KAMP_2020_S_KET_ESKETAMINE, "KET_S",
     kamp_2020_escalating_regimen(REF_WEIGHT_KG, "esketamine"),
     "arterial", "Kamp 2020"),
    ("S-ketamine, escalating racemate", KAMP_2020_S_KET_RACEMIC, "KET_S",
     kamp_2020_escalating_regimen(REF_WEIGHT_KG, "racemic"),
     "arterial", "Kamp 2020"),
    ("R-ketamine, escalating racemate", KAMP_2020_R_KET_RACEMIC, "KET_R",
     kamp_2020_escalating_regimen(REF_WEIGHT_KG, "racemic"),
     "arterial", "Kamp 2020"),
    ("S-norketamine, escalating esketamine", KAMP_2020_S_NK_ESKETAMINE, "NK_S",
     kamp_2020_escalating_regimen(REF_WEIGHT_KG, "esketamine"),
     "arterial", "Kamp 2020"),
    ("S-norketamine, escalating racemate", KAMP_2020_S_NK_RACEMIC, "NK_S",
     kamp_2020_escalating_regimen(REF_WEIGHT_KG, "racemic"),
     "arterial", "Kamp 2020"),
    ("R-norketamine, escalating racemate", KAMP_2020_R_NK_RACEMIC, "NK_R",
     kamp_2020_escalating_regimen(REF_WEIGHT_KG, "racemic"),
     "arterial", "Kamp 2020"),
    ("Hydroxynorketamine, escalating racemate", KAMP_2020_HNK_RACEMIC, "HNK",
     kamp_2020_escalating_regimen(REF_WEIGHT_KG, "racemic"),
     "arterial", "Kamp 2020"),
]


def predict(regimen, species, times, t_end):
    grid = np.linspace(0.0, t_end, 2000)
    r = L1Model(default_parameters()).simulate(regimen, t_end, t_eval=grid)
    if not r.success:
        raise RuntimeError("L1 integration failed")
    conc = r.concentration(species, "cen")
    return np.interp(times, grid, conc), grid, conc


def metrics(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    keep = (obs > 0) & np.isfinite(pred) & (pred > 0)
    obs, pred = obs[keep], pred[keep]
    if obs.size == 0:
        return None
    pe = (pred - obs) / obs * 100.0
    log_res = np.log10(pred) - np.log10(obs)
    ratio = pred / obs
    return {
        "n_points": int(obs.size),
        "MPE_percent": float(np.median(pe)),
        "MAPE_percent": float(np.median(np.abs(pe))),
        "RMSLE_log10": float(np.sqrt(np.mean(log_res ** 2))),
        "fraction_within_2fold": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
        "geometric_mean_ratio": float(10 ** np.mean(log_res)),
    }


def main():
    rows = []
    print("Quantitative prediction error of the L1 pharmacokinetic layer")
    print(f"{'curve':42s} {'n':>3s} {'MPE%':>8s} {'MAPE%':>8s} "
          f"{'RMSLE':>7s} {'F2':>6s} {'GMR':>6s}")
    for label, curve, species, regimen, site, study in CURVES:
        t_end = float(max(curve.times_h) * 1.05 + 1.0)
        pred, grid, conc = predict(regimen, species, curve.times_h, t_end)
        m = metrics(curve.concentrations, pred)
        if m is None:
            continue
        i_obs = int(np.argmax(curve.concentrations))
        i_pred = int(np.argmax(conc))
        m.update({
            "label": label, "species": species, "study": study,
            "sampling_site": site,
            "source": curve.source,
            "observed_Cmax_mgL": float(np.max(curve.concentrations)),
            "predicted_Cmax_mgL": float(np.max(conc)),
            "observed_Tmax_h": float(curve.times_h[i_obs]),
            "predicted_Tmax_h": float(grid[i_pred]),
        })
        m["Cmax_ratio"] = m["predicted_Cmax_mgL"] / m["observed_Cmax_mgL"]
        rows.append(m)
        print(f"{label:42s} {m['n_points']:3d} {m['MPE_percent']:+8.1f} "
              f"{m['MAPE_percent']:8.1f} {m['RMSLE_log10']:7.2f} "
              f"{m['fraction_within_2fold']:6.2f} "
              f"{m['geometric_mean_ratio']:6.2f}")

    all_pe, all_log = [], []
    for label, curve, species, regimen, site, study in CURVES:
        t_end = float(max(curve.times_h) * 1.05 + 1.0)
        pred, _, _ = predict(regimen, species, curve.times_h, t_end)
        obs = np.asarray(curve.concentrations, dtype=float)
        keep = (obs > 0) & (pred > 0)
        all_pe.extend(((pred[keep] - obs[keep]) / obs[keep] * 100.0).tolist())
        all_log.extend((np.log10(pred[keep]) - np.log10(obs[keep])).tolist())
    all_pe = np.array(all_pe)
    all_log = np.array(all_log)
    ratio = 10 ** all_log
    pooled = {
        "n_points": int(all_pe.size),
        "n_curves": len(rows),
        "MPE_percent": float(np.median(all_pe)),
        "MAPE_percent": float(np.median(np.abs(all_pe))),
        "RMSLE_log10": float(np.sqrt(np.mean(all_log ** 2))),
        "fraction_within_2fold": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
        "geometric_mean_ratio": float(10 ** np.mean(all_log)),
    }
    print(f"\nPooled over {pooled['n_curves']} curves "
          f"({pooled['n_points']} points): "
          f"MPE {pooled['MPE_percent']:+.1f}%, "
          f"MAPE {pooled['MAPE_percent']:.1f}%, "
          f"{100*pooled['fraction_within_2fold']:.0f}% within two-fold")

    venous = [r for r in rows if r["sampling_site"] == "venous"]
    arterial = [r for r in rows if r["sampling_site"] == "arterial"]
    by_site = {
        "venous": {"n_curves": len(venous),
                   "median_MAPE_percent": float(np.median(
                       [r["MAPE_percent"] for r in venous])) if venous else None,
                   "median_Cmax_ratio": float(np.median(
                       [r["Cmax_ratio"] for r in venous])) if venous else None},
        "arterial": {"n_curves": len(arterial),
                     "median_MAPE_percent": float(np.median(
                         [r["MAPE_percent"] for r in arterial]))
                     if arterial else None,
                     "median_Cmax_ratio": float(np.median(
                         [r["Cmax_ratio"] for r in arterial]))
                     if arterial else None},
    }
    print(f"  venous curves: median Cmax ratio "
          f"{by_site['venous']['median_Cmax_ratio']:.2f}; "
          f"arterial curves: median Cmax ratio "
          f"{by_site['arterial']['median_Cmax_ratio']:.2f}")

    out = {
        "note": ("No pharmacokinetic parameter was estimated from these "
                 "curves; the parameters derive from published population "
                 "analyses, so the errors reported here are prediction errors "
                 "for independent data. The model represents a venous central "
                 "compartment, so arterially sampled curves are expected to be "
                 "underpredicted around the peak."),
        "per_curve": rows,
        "pooled": pooled,
        "by_sampling_site": by_site,
    }
    (RESULTS / "pk_goodness_of_fit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"  saved {RESULTS / 'pk_goodness_of_fit.json'}")


if __name__ == "__main__":
    main()
