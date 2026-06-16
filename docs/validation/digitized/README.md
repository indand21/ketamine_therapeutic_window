# Digitized Reference Data — Kamp 2020 & Hasan 2021

Reference PK data extracted from published ketamine figures/tables, for L1 PK
model validation. Produced by reproducible pixel digitization (PIL+numpy):
`scripts/autodigitize.py` (Hasan) and `scripts/kamp_digitize.py` (Kamp).

## Sources

| Key | Citation | DOI |
|-----|----------|-----|
| Kamp 2020 | Kamp J, et al. *Br J Anaesth* 2020;125(5):750-761 | 10.1016/j.bja.2020.06.067 |
| Hasan 2021 | Hasan M, et al. *Anesthesiology* 2021;135(2):326-339 | 10.1097/ALN.0000000000003829 |

> **Filename warning:** the PNGs in `figures_for_digitization/` are mislabeled.
> The Kamp 10-panel concentration figure (**Fig 1**, journal p754) is in
> `kamp_table1_popPK_params_p5.png`; `kamp_fig4_model_simulations_p6.png` is
> actually the Fig 2 *schematic diagram* (not curves). Verify by eye before use.

## Files

### Tabulated parameters — exact (highest reliability)
- **`hasan2021_table1_pk_parameters.csv`** — Hasan Table 1, long format: Cmax,
  Vdss, t½, CL, HNK/KET AUC ratio, renal excretion, oral F (R/S; IV 5 mg + oral).

### Hasan Fig 1 concentration-time curves — IV 5 mg (bold red line), ng/mL
- **`hasan_{S,R}_ketamine_iv5mg.csv`** — 0–48 h.

### Kamp Fig 1 concentration-time curves — nmol/mL, 0–5 h
10 panels (a–j). Naming: `kamp_<panel>_<species>_<formulation>.csv`.

> **Regimen (important):** Kamp 2020 administered an **escalating 3-step i.v.
> infusion over 180 min** (not a bolus) — which is why every panel peaks at ~3 h.
> Exact rates (Olofsen 2021, same Leiden dataset, open-access PMC8258972), three
> 60-min steps doubling each step:
> | Step | Racemic | Esketamine (pure S) |
> |------|---------|---------------------|
> | 0–60 min   | 0.28 mg/kg/h | 0.14 mg/kg/h |
> | 60–120 min | 0.57 mg/kg/h | 0.28 mg/kg/h |
> | 120–180 min| 1.14 mg/kg/h | 0.57 mg/kg/h |
>
> Totals @70 kg ≈ 140 mg (racemic) / 70 mg (esketamine). Implemented as
> `kamp_2020_escalating_regimen()` in `src/validation/l1_pk_validation.py`; the
> calibrator builds it automatically for these curves (model Tmax = 3.0 h ✓).
| File | Species | Formulation | y_max |
|------|---------|-------------|-------|
| `kamp_a_S-ketamine_esketamine`    | S-ketamine    | esketamine | 1.6 |
| `kamp_b_S-ketamine_racemic`       | S-ketamine    | racemic    | 1.6 |
| `kamp_c_R-ketamine_racemic`       | R-ketamine    | racemic    | 1.6 |
| `kamp_d_S-norketamine_esketamine` | S-norketamine | esketamine | 1.4 |
| `kamp_e_S-norketamine_racemic`    | S-norketamine | racemic    | 1.4 |
| `kamp_f_R-norketamine_racemic`    | R-norketamine | racemic    | 1.4 |
| `kamp_g_S-DHNK_esketamine`        | S-DHNK        | esketamine | 0.20 |
| `kamp_h_S-DHNK_racemic`           | S-DHNK        | racemic    | 0.20 |
| `kamp_i_R-DHNK_racemic`           | R-DHNK        | racemic    | 0.20 |
| `kamp_j_HNK-total_racemic`        | total HNK     | racemic    | 0.5 |

## Method

For each panel the target-colored curve is extracted column-by-column, taking
the **centroid of the longest contiguous colored run** to isolate the bold mean
line from the thin SD-whisker error bars; then resampled to a clean time grid.

**Hasan** (`autodigitize.py`): axes + tick marks auto-detected per panel; y
calibrated from tick marks (0 at x-axis, 20 ng/mL at topmost tick).

**Kamp** (`kamp_digitize.py`): Fig 1 is a uniform 4×3 ggplot template, so axis
geometry was measured once by fitting evenly-spaced tick ladders and hard-coded
as absolute pixel coordinates (columns 0→5 h; rows y=0 px; constant 394 px plot
height; y_max per row). This is far more robust than per-panel detection, which
fails on the dense, low-contrast R-isomer / metabolite panels.

## Validation (built-in self-checks → `manifest.json`, `kamp_manifest.json`)

**Hasan** — extracted upper SD-whisker envelope matches published *mean + SD*:
S-ket 20.13 vs 20.1; R-ket 21.27 vs 21.4 ng/mL (~1%). Mean-line peak ~10–20%
below published Cmax (apex/infusion-phase artifact); decay phase reliable.

**Kamp** — all 10 panels digitized. The 7 panels with a clear, readable peak
(a,b,c,d,e,g,h) match the by-eye peak to **within ±5%, most < ±1%** (e.g.
S-ket(esk) 1.076 vs 1.08 @ 3.0 h; S-DHNK(esk) 0.063 vs 0.063). The faint
R-isomer / total-HNK panels (f +7%, i +15%, j +19%) differ from rough by-eye
peak estimates that are themselves ±15% uncertain. No tabulated Cmax exists for
these simulated curves, so the eyeball read is the only reference. Per-panel
numbers in `kamp_manifest.json`; audit overlays in `_debug/kamp_*_overlay.png`.

## Caveats

- Curve apex / IV-infusion rise (Hasan 0–0.5 h) is the least reliable region;
  decay phases (which drive t½) are well captured.
- Kamp panel j (total HNK) covers ~half the x-columns (curve sits near y=0 early,
  overlapping the baseline) — early near-zero points are sparse; peak is reliable.
- Prefer the **table CSV** for quantitative anchors (Cmax, t½); use curve CSVs
  for time-course/shape.
- mistral-OCR of the tables was unavailable (401); Hasan Table 1 was transcribed.

## Provenance / regeneration
`python scripts/autodigitize.py` and `python scripts/kamp_digitize.py`
(cwd-independent; both chdir to repo root). Diagnostics in `_debug/`.
