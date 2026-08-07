# Documentation

Design documents, parameter provenance, and validation data for the multiscale
QSP ketamine model.

| File | Contents |
|---|---|
| `WP1_TechSpec_PBPK_PK_PGx.md` | Technical specification for layer 1 (compartmental structure, state equations, species, dosing input, pharmacogenetic covariate, verification checks). Referenced from `src/l1_pk/`, `src/pgx/`, `src/verification/`. |
| `WP2_TechSpec_L2_Occupancy_L3a_SD.md` | Technical specification for layer 2 (state-dependent NMDA receptor open-channel block) and layer 3a (spreading depolarisation). |
| `Calibration_Strategy.md` | Hierarchical, layer-wise calibration strategy and the rationale for not attempting a single global fit. |
| `L1_Parameter_Provenance.md` | Provenance of the layer 1 parameters. **Superseded in part**: the disposition parameters are now estimated by `scripts/run_l1_calibration.py` rather than transcribed from a population analysis of intranasal esketamine. See `L1_Calibration.md`. |
| `L1_Calibration.md` | The current calibration: design, estimates, uncertainty, and prediction error on held-out data. This is the authoritative record for the layer 1 parameter values in `src/l1_pk/config.py`. |
| `L2_Parameter_Provenance.md` | Provenance of the receptor kinetic parameters. |
| `L3a_Parameter_Provenance.md` | Provenance of the spreading depolarisation parameters. |
| `L3b_Parameter_Provenance.md` | Provenance of the NMDA-hypofunction parameters. |
| `Identifiability.md` | Structural identifiability of the injury layer: which coefficient combinations are estimable, and the critical toxic weight at which the dose-response window appears. |
| `Risk_Assessment_ToxicArm_AntiCircularity.md` | Risk assessment for the toxic arm, including the anti-circularity protocol used to check that the window is mechanism-driven rather than imposed. |
| `validation/digitized/` | Concentration-time data digitised from published figures (Hasan 2021; Kamp 2020), with the digitisation method, manifests, and self-checks. |

## A note on scope

Several downstream coefficients of the protective and toxic layers are not
identifiable from currently available data. Those outputs are dimensionless and
are intended to be read qualitatively. `Identifiability.md` states precisely
which combinations are estimable and which are not, and what the qualitative
conclusions do and do not depend on.
