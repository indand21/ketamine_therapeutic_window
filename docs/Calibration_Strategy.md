# Calibration Strategy — Hierarchical Bayesian Inference Workflow

**Project:** QSP Framework for the Neuroprotective Ketamine Dose Window
**Parent protocol:** `QSP_Ketamine_Neuroprotection_Protocol_v1.0.md` (§6, §5, §7, §10)
**Scope:** L1 (WP1) → L2/L3a (WP2) → L3b/L4 (WP3); focus on WP1→WP2 hand-off
**Status:** Draft v0.1 (strategy document)

---

## 1. Principle: hierarchical, layer-wise — not a single global fit

Calibration follows a **hierarchical, layer-wise Bayesian** strategy
(Protocol §6). A single global fit is explicitly rejected because (i) the
layers are observed by different data modalities (plasma PK vs ECoG SD burden
vs preclinical histopathology), and (ii) the toxic arm is poorly identifiable
and must not be allowed to contaminate the well-constrained PK layer. The
workflow propagates **full posteriors** between layers rather than point
estimates, so parameter uncertainty flows coherently into the window.

```
[Step 1] PK first        L1 popPK  →  posterior p(θ_PK | C_obs)
   │  (carry full posterior, not point estimates)
[Step 2] PK–SD           L2→L3a map fit to ECoG SD burden,
   │                     PK fixed/informed by Step-1 posterior
[Step 3] Toxic arm       L3b with strongly informative preclinical priors
   │                     (poorly identifiable; uncertainty propagated)
[Step 4] Joint refine    limited joint update where data permit; shrinkage controlled
```

---

## 2. Step 1 — PK first (WP1 / L1)

**Method (Protocol §6.1):** Bayesian NLME (population PK) for L1 from
concentration data. Estimate the parameter vector `θ_PK` = {enantiomer/
metabolite CL, V, BBB transfer (`CL_in`, `CL_out`, `Kp,uu,brain`), CYP2B6/3A4
covariate effects `θ_2B6(g)`, `φ_k`} (WP1 TechSpec §3–4).

**Posterior carried forward (load-bearing):**

$$p(\theta_{PK}\mid C_{obs}) \;\propto\; p(C_{obs}\mid \theta_{PK})\,p(\theta_{PK})$$

Full posterior — not the MAP/EBE point — is retained. Pipelines: **VEM-NLME /
EBE** for individual-level conditioning; population-level draws retained for
forward propagation (Protocol §6.1).

**Engines (Protocol §6, §11):** nlmixr2 / Monolix (NLME layer); or
NumPyro / PyMC / Stan; **mrgsolve** for fast QSP simulation in R.

**Key exported quantity for WP2:** posterior predictive trajectories of the
PD driver `C_ecf^{s}(t)` per regimen and genotype, with their uncertainty.

---

## 3. Step 2 — PK–SD map (WP2 / L2→L3a): how WP1 posteriors inform it

This is the central hand-off. The WP2 deliverable (the **PK→SD map**,
`B_pyr → {λ_SD, D_SD, τ_rec}`) is calibrated to **ECoG SD-burden** data with
**PK fixed/informed by the Step-1 posteriors** (Protocol §6.2).

### 3.1 Mechanism of information transfer

The L2→L3a parameters `θ_SD` = {`k_on^{S,R}`, `k_off`, `P_open` params
`K_Glu,p`, `σ_p`; SD gains `κ`, `D_SD`, `τ_rec` mapping} are inferred
**conditional on** the PK posterior. Two coupling modes are supported:

- **Fixed-informed (primary):** sample `θ_PK^{(m)} ∼ p(θ_PK|C_obs)`; for each
  draw, simulate `C_ecf^{(m)}(t)` (WP1) and fit `θ_SD` to ECoG. This is a
  **cut / plug-in** posterior that prevents SD data from feeding back and
  distorting the well-identified PK layer.
- **Informative-prior (secondary):** use the Step-1 marginal posterior as a
  prior on `θ_PK` within the WP2 model, allowing limited update only where SD
  data are PK-informative (reserved for Step 4).

### 3.2 Hierarchical likelihood (PK-conditioned)

$$p(\theta_{SD}\mid \text{ECoG})\;\propto\; \underbrace{\Big[\!\int p(\text{ECoG}\mid \theta_{SD},C_{ecf}(\theta_{PK}))\,p(\theta_{PK}\mid C_{obs})\,d\theta_{PK}\Big]}_{\text{PK uncertainty marginalized in}}\;p(\theta_{SD})$$

The inner integral marginalizes PK uncertainty into the SD likelihood, so the
PK→SD map's credible intervals correctly inherit WP1 PK uncertainty. In
practice this is evaluated by Monte Carlo over the `θ_PK^{(m)}` draws.

### 3.3 Observation model

ECoG SD-burden = `∫ λ_SD·D_SD dt` (WP2 TechSpec §3.4) with an appropriate
count/duration error model. Genotype-stratified subjects link to PK through
`θ_2B6(g)` so that PGx variability in `C_ecf` propagates into SD predictions.

---

## 4. Step 3 — Toxic arm (L3b), stated up front

L3b parameters are **poorly identifiable from human data** (Protocol §6.3,
§13). They are fit with **strongly informative priors** derived from
preclinical allometric/PD scaling, and that uncertainty is **propagated
explicitly**. This is declared up front because it is **load-bearing**: the
**upper bound of the window will partly reflect priors**, and the protocol
owns that. Detail in the Risk Assessment doc.

---

## 5. Step 4 — Joint refinement

A **limited joint update** where data permit, with **shrinkage controlled**
(Protocol §6.4). The PK layer remains anchored by Step-1 posteriors; joint
moves are restricted to parameters the combined data can identify, monitored
via the identifiability diagnostics (Protocol §7).

---

## 6. Qualification linkage (Protocol §10)

- **Internal:** posterior predictive checks; cross-validation of the PK–SD map.
- **External:** held-out ECoG/PK datasets; the **~1.15 mg/kg/h** SD-suppression
  behaviour must be **reproduced, not fitted** — predicted by the calibrated
  mechanism (Carlson 2018/2019; Sánchez-Porras 2022; Reinhart 2024).
- **Identifiability gating (Protocol §7):** structural identifiability checked
  before fitting; practical identifiability (profile likelihood, FIM) reported
  per layer, labelling each window boundary **data-constrained vs
  prior-constrained**.

---

## 7. Provenance and reproducibility (Protocol §11, §14)

Seeded reproducibility; parameter/run provenance tracking; transparent
reporting of prior provenance for non-identifiable (toxic-arm) parameters.

---

## 8. Literature anchors (Protocol §15)

PK: published ketamine/norketamine/HNK popPK; CYP2B6 PGx. PK–SD: Carlson
2018/2019, Sánchez-Porras, Reinhart 2024, Hertle/COSBID 2012 (human ECoG).
Toxic-arm priors: Olney NRHypo literature. Re-verify against primary sources
during WP1 (Protocol §248).
