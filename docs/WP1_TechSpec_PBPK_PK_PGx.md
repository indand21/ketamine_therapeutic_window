# WP1 — Technical Specification: L1 PBPK/PK + Pharmacogenetics

**Project:** QSP Framework for the Neuroprotective Ketamine Dose Window
**Parent protocol:** `QSP_Ketamine_Neuroprotection_Protocol_v1.0.md` (§3.2, §5, §6.1, §8)
**Work package:** WP1 (Months 0–6)
**Deliverable:** Calibrated enantiomer/metabolite PK with full posteriors; VP PK layer
**Status:** Draft v0.1 (technical specification)

---

## 1. Purpose and scope

WP1 implements layer **L1** of the multiscale stack (Protocol §2): a reduced
PBPK / minimal physiologically-based model that delivers the pharmacodynamic
driver consumed by L2 — the **brain extracellular-fluid (ECF) concentration**
of each tracked species, *not* plasma concentration (Protocol §3.2).

The module tracks four species as separate entities:

| Symbol | Species | Role |
|---|---|---|
| `S-KET` | S-ketamine (esketamine) | High-affinity NMDAR open-channel blocker |
| `R-KET` | R-ketamine | ~3–4× lower NMDAR affinity than S (enters L2 via `k_on`) |
| `NK` | norketamine | Active metabolite (N-demethylation product) |
| `HNK` | (2R,6R)-hydroxynorketamine | NMDAR-independent; plasticity/antidepressant arm |

Keeping metabolites as separate species is a deliberate design requirement
(Protocol §3.2) so the same engine serves the perioperative/antidepressant
translations where parent and metabolite act on different downstream nodes.

---

## 2. Compartmental structure

Reduced PBPK with a dedicated **brain compartment** and explicit blood–brain
barrier (BBB) transfer. Minimum compartment set per species `s`:

- `cen` — central/plasma (dosing + sampling site)
- `per` — peripheral/tissue lumped (distribution kinetics)
- `bbb`/`brain_vasc` — brain vascular sub-compartment (BBB interface)
- `ecf` — brain extracellular fluid (**PD driver, exported to L2**)

```
[L0 dosing] → cen ⇄ per
                │
               cen ⇄ brain_vasc ⇄ ecf   (BBB transfer, asymmetric Kp,uu)
                │  metabolism (CYP)
            S-KET → NK → HNK   (R-KET → R-NK → (2R,6R)-HNK, enantioselective)
```

ECF concentration relates to total brain concentration via the unbound
brain-to-plasma partition coefficient `Kp,uu,brain`; the BBB flux uses
influx/efflux clearances `CL_in`, `CL_out` (P-gp efflux flagged as a
sensitivity extension, consistent with the L2 off-target convention).

---

## 3. State equations

### 3.1 Generic mass-balance template (Protocol §3.2)

For generic compartment `i`, species `s`, the protocol's canonical form is
preserved verbatim:

$$V_i \frac{dC_{i,s}}{dt} = \sum_j Q_{ij}\,(C_{j,s}-C_{i,s}) \;-\; \mathrm{CL}_{i,s}(\text{genotype})\,C_{i,s} \;+\; R_{\text{met},s}$$

where `V_i` = compartment volume, `Q_ij` = inter-compartmental flow,
`CL_{i,s}(genotype)` = genotype-dependent clearance (§4), and `R_met,s` =
net metabolic source/sink for species `s` in compartment `i`.

### 3.2 Species-resolved instantiation (central compartment, enantiomer e ∈ {S,R})

Parent ketamine (formation of NK is a sink for KET, source for NK):

$$V_{cen}\frac{dC_{cen}^{KET_e}}{dt} = u_e(t) + Q_{per}(C_{per}^{KET_e}-C_{cen}^{KET_e}) - CL_{in}^{e}C_{cen}^{KET_e} + CL_{out}^{e}C_{vasc}^{KET_e} - CL_{met,NK}^{e}(g)\,C_{cen}^{KET_e}$$

Norketamine (formed from parent, cleared onward to HNK):

$$V_{cen}\frac{dC_{cen}^{NK_e}}{dt} = f_m\,CL_{met,NK}^{e}(g)\,C_{cen}^{KET_e} - CL_{met,HNK}^{e}(g)\,C_{cen}^{NK_e} - CL_{other}^{NK_e}C_{cen}^{NK_e}$$

(2R,6R)-HNK (terminal tracked metabolite, NMDAR-independent):

$$V_{cen}\frac{dC_{cen}^{HNK}}{dt} = f_{m,HNK}\,CL_{met,HNK}^{R}(g)\,C_{cen}^{NK_R} - CL_{out}^{HNK}C_{cen}^{HNK}$$

`u_e(t)` is the L0 dosing input for enantiomer `e` (bolus + infusion;
racemate sets `u_S = u_R`; S-ketamine regimen sets `u_R = 0`), and `f_m` is
the fraction of parent clearance routed to N-demethylation.

### 3.3 Brain ECF (PD export to L2)

$$V_{ecf}\frac{dC_{ecf}^{s}}{dt} = CL_{in}^{s}C_{vasc}^{s} - CL_{out}^{s}C_{ecf}^{s}, \qquad C_{brain}^{s}\big|_{L2} \equiv C_{ecf}^{s}$$

The quantity `C_brain` referenced by the L2 kinetics (Protocol §3.3) is bound
to `C_ecf`. At pseudo-steady state `C_ecf/C_cen,u → Kp,uu,brain`.

---

## 4. CYP2B6 genetic covariate model

Metabolism (Protocol §3.2, §5): **CYP2B6** is the principal route of
N-demethylation to norketamine, with **CYP3A4 / CYP2C9** contributions and
further oxidation to HNK; clearance is **enantioselective**.

Pharmacogenetic variability enters as a multiplicative covariate on the
N-demethylation clearance for genotype `g`:

$$CL_{met,NK}^{e}(g) = CL_{met,NK}^{e,\,*1/*1}\cdot \theta_{2B6}(g)\cdot\Big(1+\textstyle\sum_k \phi_k\,\mathbb{1}[\text{covariate}_k]\Big)$$

- Variant set: **CYP2B6\*6** (defining SNPs **516G>T [rs3745274]**,
  **785A>G [rs2279343]**) and related reduced-function alleles, encoded as
  diplotype → activity score → `θ_2B6(g)` (Protocol §3.2).
- Reduced-function diplotypes (`*6/*6`) decrease `θ_2B6` → slower parent
  clearance → higher/longer `C_ecf` → leftward shift of the dosing axis for
  a given exposure (carried into the VP, Protocol §8).
- CYP3A4/CYP2C9 contributions enter as additional `φ_k` covariate terms on
  the residual (non-2B6) clearance fraction.

### 4.1 Indian-population allele frequencies (VP layer)

The virtual-population PK layer samples diplotypes from **Indian
pharmacogenomic datasets / 1000G South Asian** allele frequencies
(Protocol §5, §8), paralleling the CYP3A5/tacrolimus clearance logic. The
diplotype sampler produces `θ_2B6(g)` per virtual subject; these feed the VP
PK simulations and, downstream, the population window distribution.

---

## 5. Inputs, outputs, and interfaces

**Inputs (from L0, Protocol §3.1):** bolus (mg/kg), infusion (mg/kg/h),
loading+maintenance schedules, enantiomer composition (R/S ratio).

**Outputs (to L2, Protocol §3.3):** time courses `C_ecf^{S-KET}(t)`,
`C_ecf^{R-KET}(t)`, `C_ecf^{NK}(t)`, `C_ecf^{HNK}(t)`; plus plasma profiles
for calibration against observed concentration data.

**Parameters & sources (Protocol §5):** enantiomer/metabolite CL, V, BBB
transfer; CYP2B6/3A4 covariates — from published ketamine/norketamine/HNK
popPK (anesthesia, analgesia, depression), enantioselective PK studies, and
CYP2B6 PGx literature.

---

## 6. Calibration hook (see Calibration Strategy doc, Protocol §6.1)

WP1 is the **PK-first** step: Bayesian NLME (population PK) estimates L1
parameters from concentration data; **full posteriors** are carried forward
(VEM-NLME / EBE pipelines). These posteriors fix/inform PK when WP2
calibrates the PK→SD map. Engines: nlmixr2 / Monolix (NLME); NumPyro/PyMC/Stan;
mrgsolve for fast simulation (Protocol §6, §11).

---

## 7. Verification (Protocol §10)

- Mass conservation across compartments (closed-system check, zero clearance).
- Unit consistency; ODE solver convergence (tolerance sweep).
- Regression tests vs published popPK profiles; reproducible, seeded runs.
- Qualification target downstream: enantioselective clearance ratios and
  Kp,uu,brain consistent with literature ranges.

---

## 8. Literature anchors (Protocol §15)

Published ketamine/norketamine/HNK popPK; enantioselective PK studies;
CYP2B6 PGx literature; Indian pharmacogenomic datasets / 1000G South Asian.
*Provenance note (Protocol §248): clinical anchors to be re-verified against
primary sources during WP1.*
