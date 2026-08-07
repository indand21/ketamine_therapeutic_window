# Initial Risk Assessment — Toxic Arm (L3b) Identifiability & Anti-Circularity

**Project:** QSP Framework for the Neuroprotective Ketamine Dose Window
**Parent protocol:** `QSP_Ketamine_Neuroprotection_Protocol_v1.0.md` (§7, §6.3, §13, §0)
**Scope:** Identifiability risks of the toxic arm and the emergence-verification safeguard
**Status:** Draft v0.1 (risk assessment)

---

## 1. The dominant risk and why it exists

The defining methodological risk of the program is that the **therapeutic
window must be an *emergent property* of the mechanism, not an imposed
functional form** (Protocol §0, §7). The explicit corrective is to a
parametric model in which the trough is **hard-coded** and all output panels
are **algebraically linked** (circular convergence). Two structural facts make
this hard:

1. The protective and toxic arms **move together with dose**, both driven by
   the same molecular event (NMDAR channel block) (Protocol §0).
2. The toxic arm (NRHypo) is **essentially unobservable in vivo** — no
   empirical dose-titration can separate the arms (Protocol §0).

Consequently the **upper bound of the window is partly prior-driven**, and the
model is **hypothesis-generating, not confirmatory**, for that boundary
(Protocol §6.3, §13).

---

## 2. Toxic-arm (L3b) identifiability risks

L3b is the neural-mass E–I disinhibition / NRHypo model. Block of interneuron
NMDARs `B_int` reduces inhibitory drive → pyramidal disinhibition → sustained
glutamate → cumulative injury index (Protocol §3.5):

$$T_{\text{NRHypo}} = \int_0^{t} g\big(B_{\text{int}}(\tau)\big)\,d\tau, \qquad g \text{ convex above a disinhibition threshold}$$

| # | Risk | Driver | Consequence | Mitigation |
|---|---|---|---|---|
| R1 | **Non-identifiability of `g` (disinhibition gain & convexity)** | No human in-vivo readout of NRHypo (Protocol §0, §6.3) | Upper window boundary under-determined by data | Strongly informative priors from preclinical allometric/PD scaling; propagate uncertainty (Protocol §6.3) |
| R2 | **Disinhibition threshold location** | Threshold of convex `g` sets where toxicity turns on | Trough/upper-bound shifts with prior, not data | Profile-likelihood + FIM; label boundary prior- vs data-constrained (§7) |
| R3 | **Interneuron occupancy `B_int` separation from `B_pyr`** | Shared `C_brain`; population `{k_on,k_off,P_open}` differences are the only separation lever (L2 §2.2) | If populations not distinct, two-arm behaviour collapses | In-vitro subunit-specific kinetics; sensitivity on population params |
| R4 | **Preclinical→human PD scaling** | Olney/NRHypo data are animal histopathology | Structural uncertainty in toxic magnitude | Allometric priors; quantify via GSA (§7, §13) |
| R5 | **Geriatric threshold shift** | Age-linked GABAergic reserve / NMDAR density (Protocol §8) | Window mislocated in translation arm | VP sampling of age-linked toxic threshold; treat POD as hypothesis |
| R6 | **Prior leakage into "data-driven" claims** | Informative priors dominate posterior | Over-confident window | Cut/plug-in posterior (Calibration §3.1); independence audit (§4 below) |

**Load-bearing disclosure (Protocol §6.3, §13):** because L3b parameters are
poorly identifiable from human data, the **upper bound of the window will
partly reflect priors**, and the protocol states this up front and owns it.

---

## 3. Anti-circularity / emergence-verification protocol (Protocol §7)

This is the explicit safeguard against the parametric model's failure mode.
Four pre-fit / post-fit checks are **mandatory**.

### 3.1 Structural identifiability (before fitting)
Differential-algebra / observability checks on the reduced model — confirm in
principle which parameters are recoverable from the planned observations
(Protocol §7).

### 3.2 Practical identifiability
Profile likelihood and the **Fisher information matrix**; **report which
parameters and which window boundaries are data-constrained vs
prior-constrained** (Protocol §7). Directly addresses R1–R4.

### 3.3 Global sensitivity analysis
**Sobol / Morris (SALib)** attributing the **location and width of the
window** to specific parameters, so the result is **mechanistically
attributable** rather than an artifact (Protocol §7, §11).

### 3.4 Emergence test (mandatory) — the core anti-circularity proof
Demonstrate the **U-shape is mechanism-driven, not imposed**:

- **Ablate L3b** → the **upper bound must disappear** (window opens upward).
  If it does not, the upper bound was hard-coded elsewhere — fail.
- **Perturb L2 occupancy kinetics** → the **trough must move *predictably***
  in the mechanistically expected direction (e.g., changing `k_on`/`P_open`
  shifts the dose axis of both arms coherently).
- **Independence audit** → confirm **no panel/output is an algebraic
  restatement of another** (Protocol §7) — directly rebuts the circular
  convergence failure mode (R6).

### 3.5 Falsifiability (pre-registered)
Pre-register observations that would **break** the predicted window
(Protocol §7), e.g.:
- SD suppression **failing to plateau below** the predicted toxic threshold;
- psychotomimetic burden **rising before** the predicted boundary.

---

## 4. Residual risk and governance posture

- The window's **lower bound** (insufficient protection) is anchored to
  measurable SD-burden data and the externally-reproduced ~1.15 mg/kg/h
  behaviour (Protocol §0, §10) — comparatively well constrained.
- The window's **upper bound** (NRHypo toxicity) remains the dominant residual
  risk and is reported as **partly prior-driven** (Protocol §13).
- **Perioperative/POD translation** rests on a **contested** endpoint
  (PODCAST null, dose-dependent psychotomimetic harm; pooled RR ≈ 0.71,
  esketamine ≈ 0.59); predicted effect sizes **must not exceed meta-analytic
  bounds without justification** — use the model to **test, not assert**, a
  window (Protocol §0, §13).
- **Governance (Protocol §14):** pre-registration of falsifiability criteria
  and qualification plan; transparent reporting of prior provenance for
  non-identifiable parameters; risk-based MIDD / ASME V&V-40 credibility
  assessment (Protocol §10).

---

## 5. Literature anchors (Protocol §15)

Olney et al. (NRHypo / NMDA-antagonist neurotoxicity); Avidan 2017 (PODCAST);
Møller 2024 meta-analysis; Carlson 2018/2019, Reinhart 2024, Sánchez-Porras
(protective-arm anchors). Re-verify against primary sources (Protocol §248).
