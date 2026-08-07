# WP2 — Technical Specification: L2 Occupancy + L3a SD Engine

**Project:** QSP Framework for the Neuroprotective Ketamine Dose Window
**Parent protocol:** `QSP_Ketamine_Neuroprotection_Protocol_v1.0.md` (§3.3, §3.4, §5, §6.2, §10)
**Work package:** WP2 (Months 4–12)
**Deliverable:** PK→SD map; external SD-threshold reproduction (~1.15 mg/kg/h)
**Status:** Draft v0.1 (technical specification)

---

## 1. Purpose and scope

WP2 implements **L2** (state-dependent NMDAR target engagement) and the
**L3a protective arm** (spreading-depolarization susceptibility/propagation),
and couples them into the calibratable **PK→SD map**. The PD driver is the
brain ECF concentration `C_brain` exported by WP1/L1 (Protocol §3.2–3.3).

Central design commitment (Protocol §3.3): ketamine is a **use-dependent,
voltage-dependent, trapping open-channel blocker**. Occupancy is therefore
*not* a simple `E_max` function of concentration — block requires the channel
to be **open**, which depends on local glutamate and depolarization. This
self-targeting (more block where pathology is) is mechanistically essential
and is exactly what an Emax model discards.

---

## 2. L2 — State-dependent NMDAR open-channel block

### 2.1 Governing kinetics (Protocol §3.3, preserved verbatim)

Per neuronal population `p`:

$$\frac{dB_p}{dt} = k_{on}\,C_{\text{brain}}\,P_{\text{open},p}(V,[\mathrm{Glu}])\,(1-B_p) \;-\; k_{off}\,B_p$$

- `B_p` — fractional NMDAR block (occupancy) in population `p` ∈ {`pyr`, `int`}.
- `P_open,p(V,[Glu])` — state-dependent open probability; the gate that makes
  block use-/voltage-dependent. Function of local membrane potential `V` and
  glutamate `[Glu]`, both supplied by the L3a/L3b circuit state.
- `(1−B_p)` — available (unblocked, openable) receptor fraction.
- `k_on`, `k_off` — association/dissociation rate constants; trapping block
  implies `k_off` is small while the channel is closed (occlusion).

### 2.2 Population segregation (the crux of two-arm behaviour)

The two arms emerge from **different `{k_on, k_off, P_open}` per population**
(Protocol §3.3):

| Population | Symbol | NMDAR profile | Downstream arm |
|---|---|---|---|
| Principal / pyramidal | `B_pyr` | GluN2-determined | L3a protective (→ SD) |
| Fast-spiking GABAergic interneuron | `B_int` | distinct GluN2 profile | L3b toxic (→ NRHypo) |

Distinct subunit composition (GluN2 profile) → different block sensitivity and
kinetics → different dose-sensitivities for protective vs toxic arms. WP2 owns
`B_pyr` (→ L3a); `B_int` (→ L3b) is specified here for interface completeness
and consumed in WP3.

### 2.3 Enantiomer-specific affinity and off-targets

- Enantiomer affinity (S-ketamine ≈ **3–4×** R) enters via `k_on` (Protocol
  §3.3). With WP1 tracking species separately, the effective association term
  sums enantiomer contributions:
  `k_on·C_brain → k_on^{S}·C_ecf^{S-KET} + k_on^{R}·C_ecf^{R-KET}`.
- Off-target **HCN1** is flagged as a **sensitivity-analysis extension, not
  core** (Protocol §3.3).

### 2.4 Open-probability submodel

`P_open,p(V,[Glu])` is parameterized as a separable saturating gate:

$$P_{\text{open},p}(V,[\mathrm{Glu}]) = \sigma_p(V)\cdot \frac{[\mathrm{Glu}]}{K_{Glu,p}+[\mathrm{Glu}]}$$

with `σ_p(V)` a sigmoidal voltage-relief term (Mg²⁺-block relief surrogate)
and `K_Glu,p` the population glutamate sensitivity. Couples L2 to the L3a
state variables `V` and `[Glu]`.

---

## 3. L3a — Protective arm: SD susceptibility / propagation

### 3.1 Model class (Protocol §3.4)

Two interchangeable formulations are supported (selectable; cross-checked):

1. **Ion-based Hodgkin–Huxley-type** with K⁺ / glutamate / Na⁺ dynamics
   (**Hübel–Dahlem lineage**) — full biophysical fidelity.
2. **Reduced bistable reaction–diffusion** formulation — fast surrogate
   preserving the SD bifurcation structure for optimization/RL use.

### 3.2 Biophysical state variables and parameters

Ion-based state vector (per tissue node): membrane potential `V`,
extracellular `[K⁺]_o`, intracellular `[Na⁺]_i`, extracellular glutamate
`[Glu]_o`, gating variables. Core parameter groups (Protocol §5, L3a row):

| Group | Parameters | Source |
|---|---|---|
| Ionic | Na/K conductances, pump rate `I_pump`, leak | preclinical SD electrophysiology |
| Diffusion | K⁺/glutamate diffusion `D_K`, `D_Glu` (RD term `D∇²`) | SD propagation data |
| Threshold | SD initiation threshold `K_thr`; block→threshold gain | Sánchez-Porras swine; Reinhart mouse |
| Recovery | repolarization/recovery time constant `τ_rec` | Reinhart 2024 (peri-infarct recovery) |

### 3.3 NMDAR block → SD coupling (the protective mechanism)

NMDAR block acts as a **bifurcation parameter** moving tissue away from the
SD-prone regime. `B_pyr` modulates three SD descriptors (Protocol §3.4):

- **raises** the SD initiation threshold: `K_thr(B_pyr) = K_thr^0·(1+κ·B_pyr)`;
- **shortens** the depolarization / ECoG-depression duration `D_SD(B_pyr)`;
- **accelerates** ionic recovery `τ_rec(B_pyr)` in peri-injury tissue.

### 3.4 PK→SD map (the WP2 deliverable)

The map `B_pyr → {λ_SD, D_SD, τ_rec}` yields the protective readout as the
reduction in the **SD burden integral**:

$$\text{SD burden} = \int_0^{T}\lambda_{SD}\big(B_{pyr}(t)\big)\,D_{SD}\big(B_{pyr}(t)\big)\,dt$$

`λ_SD` = SD event rate. The formal protective dose is the exposure that shifts
the bifurcation **without** crossing into the toxic regime (Protocol §3.4) —
the boundary set jointly with WP3/L3b.

---

## 4. Interfaces

**Inputs:** `C_ecf^{S-KET,R-KET,NK,HNK}(t)` from WP1/L1.
**Internal coupling:** L3a supplies `V`, `[Glu]_o` to L2's `P_open`; L2
returns `B_pyr` to L3a (and `B_int` forward to L3b/WP3).
**Outputs:** SD burden integral, `λ_SD`, `D_SD`, `τ_rec` → L4 injury (WP3);
SD-burden trajectory → calibration (ECoG) and the closed-loop PD readout (§4
of protocol).

---

## 5. Calibration & qualification hooks

- **PK–SD calibration (Protocol §6.2):** fit the L2→L3a map to **ECoG
  SD-burden** data with PK **fixed/informed by WP1 step-1 posteriors**
  (see Calibration Strategy doc).
- **External qualification (Protocol §10):** the **~1.15 mg/kg/h**
  SD-suppression behaviour (Carlson 2018/2019; Sánchez-Porras 2022; Reinhart
  2024) must be **reproduced, not fitted** — predicted by the calibrated
  mechanism as a strong qualification test.
- **Verification:** PDE/ODE solver convergence; charge conservation in the
  ion-based model; RD vs ion-based cross-validation.

---

## 6. Literature anchors (Protocol §5, §15)

In-vitro patch-clamp NMDAR block kinetics; subunit-specific affinity
literature (L2). Sánchez-Porras (swine), Reinhart 2024 (mouse), Carlson
2018/2019, Hertle/COSBID 2012 human ECoG, Hübel–Dahlem ion-based SD models
(L3a). Re-verify dose/effect anchors against primary sources (Protocol §248).
