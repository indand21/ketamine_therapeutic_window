# Ketamine therapeutic window: a multiscale QSP model

A multiscale quantitative systems pharmacology (QSP) model of ketamine that links
enantiomer-resolved pharmacokinetics to opposing protective and toxic actions at
cortical NMDA receptors, and from there to a net injury readout. The model defines
a narrow, mechanism-driven neuroprotective window and is used to explore how dose
and delivery regimen (bolus vs infusion) shape the balance of benefit and harm.

> **Scope and framing.** The pharmacokinetic layer is calibrated to digitized
> human intravenous concentration-time data and evaluated on a held-out dataset
> (`docs/L1_Calibration.md`). Several downstream (injury, occupancy, spreading
> depolarization, and psychotomimetic) coefficients cannot be estimated from
> available data, so those outputs are in **arbitrary units** and are intended to
> be read **qualitatively**. `docs/Identifiability.md` states exactly which
> coefficient combinations are estimable and what the conclusions do and do not
> depend on. The model is hypothesis-generating, not a source of quantitative
> dosing guidance.

## Model architecture

The model is a system of ordinary differential equations organized into linked layers:

| Layer | Module | Description |
|-------|--------|-------------|
| L1 | `src/l1_pk/` | Four-compartment PK of S-/R-ketamine, norketamine, and (2R,6R)-hydroxynorketamine, with blood-brain transfer |
| L2 | `src/l2_occupancy/` | State-dependent NMDA receptor open-channel occupancy in pyramidal and interneuron populations |
| L3a | `src/l3a_sd/` | Protective layer: pyramidal occupancy raises the spreading depolarization threshold |
| L3b | `src/l3b_nrhypo/` | Toxic layer: interneuron occupancy above threshold accumulates NMDA-hypofunction injury |
| L4/L5 | `src/l4_l5/` | Net injury integration and clinical constraints (psychotomimetic burden, MAP) |
| PGx / VPop / GSA | `src/phase6/`, `src/pgx/` | CYP2B6 pharmacogenetics, virtual population, sensitivity, emergence, identifiability |

Supporting modules: `src/validation/` (PK structural validation against published
targets), `src/calibration/` (L1 calibration to digitized data), `src/ingestion/`
(digitized concentration-time data loaders), `src/verification/` (numerical checks).

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

## Reproducing the results

All analyses read the current (corrected) parameters from the source code and write
raw outputs to `results/` (CSV/JSON) and figures to `figures/` (PNG + TIFF, 300 dpi).
Both directories are committed as a convenience snapshot so the outputs can be inspected
without re-running, and both are fully regenerable by the pipeline below. Run from the
repository root with `PYTHONPATH=.`:

```bash
# 1. Calibrate the pharmacokinetic layer to digitized human data (slow: the
#    bootstrap dominates)                       ->  results/l1_calibration.json
PYTHONPATH=. python scripts/run_l1_calibration.py

# 2. Core results: secondary PK parameters, dose-response window, PODCAST arms,
#    regimen, enantiomer, virtual population, CYP2B6, emergence, overlays
PYTHONPATH=. python scripts/regenerate_all.py

# 3. Identifiability of the injury layer and the critical toxic weight
PYTHONPATH=. python scripts/run_identifiability.py

# 4. Global variance-based (Sobol) sensitivity, with convergence sequence and
#    second-order indices                                 ->  results/sobol.json
PYTHONPATH=. python scripts/run_sobol.py

# 5. Prediction error against every digitized curve
PYTHONPATH=. python scripts/run_pk_goodness_of_fit.py

# 6. Design of the study that would calibrate what remains unmeasured
PYTHONPATH=. python scripts/run_prospective_design.py

# 7. Publication figures  ->  figures/  (reads results/)
PYTHONPATH=. python scripts/make_figures.py
```

The model's predictions are then tested against human pharmacodynamic estimates that
took no part in building it (Olofsen et al., Anesthesiology 2022;136:792-801, from the
same volunteers that supplied the calibration data). These run on the model as it
stands, without refitting anything:

```bash
# transfer-clearance scale factor reproducing the measured t1/2ke0 of 8.3 min
PYTHONPATH=. python scripts/calibrate_bbb_speed.py

# regimen comparison and dose response under the human-constrained settings
PYTHONPATH=. python scripts/run_human_constrained.py

# S:R potency sweep spanning the published human estimates
PYTHONPATH=. python scripts/run_enantiomer_ratio.py

# the trial arms evaluated across that sweep
PYTHONPATH=. python scripts/run_podcast_robustness.py
```

Two switches control them, both defaulting to the original behaviour so the main
analysis reproduces unchanged: `Clearances.bbb_speed_factor`, which scales all four
blood-brain transfer clearances and leaves the unbound partition coefficient
unchanged, and `ClinicalParams.psych_model`, which selects the illustrative linear
ramp or the published sigmoid.

Step 2 onwards read the calibrated parameters from `src/l1_pk/config.py`, which
step 1 produces. Step 7 depends on the preceding steps having written the
corresponding `results/` files. `scripts/run_robustness.py` is retained but
superseded by `run_identifiability.py`, which answers the same question exactly
rather than by sampling.

### Figure numbering

Figures are numbered in order of first citation, so Fig. 4 is the identifiability
figure and Fig. 5 the regimen figure. `scripts/make_figures.py` writes them with the
units and spelling the target journal asks for (mg kg^-1 rather than mg/kg, UK
spelling).

## Tests

```bash
PYTHONPATH=. python -m pytest
```

The suite (141 tests) covers mass conservation, numerical stability, each model
layer, PK validation, calibration, and the virtual-population / sensitivity
machinery.

## Data

`docs/validation/digitized/` contains concentration-time data digitized from published
figures (Hasan 2021; Kamp 2020) used for L1 validation and calibration, with the
digitization method and self-checks documented in that folder's `README.md`.

## Repository layout

```
config/                    L1 parameter reference copy (YAML; config.py is authoritative)
docs/                      technical specifications, parameter provenance, calibration
                           and identifiability records (see docs/README.md)
docs/validation/digitized/ digitized published PK data + provenance
scripts/                   reproduction pipeline (calibration, results, identifiability,
                           sensitivity, goodness of fit, study design, figures)
src/                       model source (layers L1-L5, PGx, VPop, GSA, validation)
tests/                     unit and property tests
```

## Citation

This code accompanies the manuscript "The delivery regimen, not the dose: a ketamine
systems pharmacology model tested against independent human pharmacodynamic data"
(Mishra N, Anil A, Archana AS, Keerthy SU, Srinivasan A), under submission. Please
cite the publication once available; until then, cite this repository.

## License

Released under the MIT License (see `LICENSE`).
