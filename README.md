# Ketamine therapeutic window: a multiscale QSP model

A multiscale quantitative systems pharmacology (QSP) model of ketamine that links
enantiomer-resolved pharmacokinetics to opposing protective and toxic actions at
cortical NMDA receptors, and from there to a net injury readout. The model defines
a narrow, mechanism-driven neuroprotective window and is used to explore how dose
and delivery regimen (bolus vs infusion) shape the balance of benefit and harm.

> **Scope and framing.** Several downstream (injury, occupancy, spreading
> depolarization, and psychotomimetic) parameters are not calibrated against
> clinical outcome data, so those outputs are in **arbitrary units** and are
> intended to be read **qualitatively**. The model is hypothesis-generating, not a
> source of quantitative dosing guidance.

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
Both directories are git-ignored because they are fully regenerable. Run from the
repository root with `PYTHONPATH=.`:

```bash
# 1. Core results: PK validation, dose-response window, PODCAST arms, regimen,
#    enantiomer, virtual population, CYP2B6, emergence, digitized overlays
PYTHONPATH=. python scripts/regenerate_all.py

# 2. Global variance-based (Sobol) sensitivity analysis  ->  results/sobol.json
PYTHONPATH=. python scripts/run_sobol.py

# 3. Robustness of the window to the uncalibrated injury parameters  ->  results/robustness.json
PYTHONPATH=. python scripts/run_robustness.py

# 4. Publication figures  ->  figures/  (reads results/)
PYTHONPATH=. python scripts/make_figures.py
```

Step 4 depends on steps 1-3 having produced the corresponding `results/` files.

## Tests

```bash
PYTHONPATH=. python -m pytest
```

The suite (141 tests) covers mass conservation, numerical stability, each model
layer, PK structural validation, calibration, and the virtual-population / sensitivity
machinery.

## Data

`docs/validation/digitized/` contains concentration-time data digitized from published
figures (Hasan 2021; Kamp 2020) used for L1 validation and calibration, with the
digitization method and self-checks documented in that folder's `README.md`.

## Repository layout

```
config/                    default L1 parameter reference (YAML)
docs/validation/digitized/ digitized published PK data + provenance
scripts/                   reproduction pipeline (regenerate, sobol, robustness, figures)
src/                       model source (layers L1-L5, PGx, VPop, GSA, validation)
tests/                     unit and property tests
```

## Citation

This code accompanies a manuscript describing the model (in preparation). Please cite
the corresponding publication when using this software.

## License

Released under the MIT License (see `LICENSE`).
