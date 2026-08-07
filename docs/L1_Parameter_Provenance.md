# L1 PBPK Parameter Provenance

> **PARTLY SUPERSEDED.** The *disposition* parameters described below (central
> and peripheral volume, intercompartmental flow, total clearance, and the
> metabolite clearances) are no longer used. They were taken from a population
> analysis of **intranasal** esketamine, whose clearance and volume are apparent
> values inflated by bioavailability; checked against intravenous
> concentration-time data, that parameterisation underpredicted plasma ketamine
> by roughly 40% and the metabolites by an order of magnitude. Those parameters
> are now estimated by `scripts/run_l1_calibration.py`. See
> **`L1_Calibration.md`**, which is authoritative for the values in
> `src/l1_pk/config.py`.
>
> This file is retained deliberately, as the record of where the earlier values
> came from and why they were wrong. The blood-brain transfer and
> hydroxynorketamine sections below remain current, since those parameters are
> still fixed rather than estimated.


**Date:** 2026-05-30
**Model:** L1 4-compartment reduced PBPK (cen / per / brain_vasc / ecf)
**Species tracked:** S-KET, R-KET, NK_S, NK_R, (2R,6R)-HNK
**Reference subject:** 70 kg healthy adult, CYP2B6\*1/\*1 (wild-type)

## Summary of Literature Sources

| Ref | Citation | N | Design | Key Parameters |
|-----|----------|---|--------|----------------|
| 1 | Kamp et al. 2020, Br J Anaesth (PMID 32838982) | 20 | IV crossover (esketamine + racemic) | 7-comp model, all species enantio-resolved |
| 2 | Hasan et al. 2021, Anesthesiology (PMID 34019627) | 15 | IV 5mg + oral PR 10–80mg | Vdss, CL, t½, F — all enantio-resolved |
| 3 | Perez-Ruixo et al. 2021, Clin Pharmacokinet (PMID 33128208) | 820 | 12 trials (IN/IV/oral esketamine) | CL=114 L/h, Vss=752 L, FRn=54%, F=18.6% |
| 4 | Weiss & Siegmund 2022, Clin Pharmacol Drug Dev (PMID 34265182) | 15 | Re-analysis of Hasan data | Steady-state HNK:KET ratios (46× R, 14× S) |
| 5 | Li et al. 2015, Br J Clin Pharmacol (PMID 25702819) | 49 | CYP2B6\*6 genotype effect on CL | CL by genotype (\*1/\*1: 68.1, \*1/\*6: 40.6, \*6/\*6: 21.6 L/h) |
| 6 | Moaddel et al. 2023, iScience (PMID 38162029) | 9 | Paired CSF+plasma PK | Only human study with CSF/plasma KET, NK, HNK |
| 7 | Schmidt & Holzgrabe 2024, Eur J Pharm Sci (PMID 37979888) | — | In vitro binding | fu S-KET ~65%, R-KET ~71% |

## Parameter-by-Parameter Derivation

### Volumes

#### V_cen (Central / Plasma Volume)

| Property | Value |
|----------|-------|
| **New value** | 45.0 L |
| **Old value** | 30.0 L |
| **Source** | Perez-Ruixo 2021, inferred |
| **Derivation** | Perez-Ruixo reports Vc/F for oral esketamine. With F ≈ 0.186, Vc/F ≈ 200–300 L implies Vc ≈ 37–56 L. Midpoint: 45 L. Consistent with plasma volume (~3 L) + rapidly equilibrating well-perfused tissues. |
| **Uncertainty** | **MODERATE** — inferred from apparent volume, not directly reported. Sensitive to F estimate. |

#### V_per (Peripheral / Tissue Volume)

| Property | Value (S) | Value (R) |
|----------|-----------|-----------|
| **New value** | 707.0 L | ~593 L (scaled) |
| **Old value** | 120.0 L (both) |
| **Source** | Hasan 2021 + Perez-Ruixo 2021 |
| **Derivation** | V_per = Vdss − V_cen. S-KET: Vdss = 752 L (Perez-Ruixo) − 45 L = 707 L. R-KET: Vdss scaled from Hasan S:R ratio (5.6/6.6 ≈ 0.848) → 752 × 0.848 = 638 L; V_per_R = 638 − 45 = 593 L. Note: the config.py uses a single V_per for both enantiomers (707 L); R-KET shares the S-KET distribution parameters. |
| **Uncertainty** | **HIGH** — Vdss from Hasan (per-kg, N=15) differs substantially from Perez-Ruixo (population, N=820). R-KET Vdss is scaled, not directly measured. |

#### V_vasc (Brain Vascular Volume)

| Property | Value |
|----------|-------|
| **New value** | 0.05 L (unchanged) |
| **Old value** | 0.05 L |
| **Source** | Anatomical constant |
| **Derivation** | Brain vascular volume ≈ 50 mL (5% of 1.4 kg brain). |
| **Uncertainty** | **LOW** — well-established anatomical value. |

#### V_ecf (Brain Extracellular Fluid Volume)

| Property | Value |
|----------|-------|
| **New value** | 0.15 L |
| **Old value** | 0.20 L |
| **Source** | Anatomical constant |
| **Derivation** | Brain ECF ≈ 10–15% of brain volume. For 1.4 kg brain: 140–210 mL. Using 150 mL (0.15 L). |
| **Uncertainty** | **LOW-MODERATE** — anatomical range; 0.15 L is conservative midpoint. |

### Flows

#### Q_per (Central ↔ Peripheral Distribution Flow)

| Property | Value |
|----------|-------|
| **New value** | 120.0 L/h |
| **Old value** | 60.0 L/h |
| **Source** | Derived from distribution half-life |
| **Derivation** | Distribution t½,α ≈ 0.1–0.2 h for ketamine (rapid initial distribution). Using t½,α = 0.15 h: Q_per ≈ 0.693 × V_cen × V_per / [t½,α × (V_cen + V_per)] ≈ 196 L/h. Capped to 120 L/h for physiological plausibility (below cardiac output ~330 L/h, above hepatic flow ~90 L/h). |
| **Uncertainty** | **HIGH** — not directly reported in any popPK model. Derived from first principles with physiological constraints. |

### BBB Transfer Clearances

All four BBB transfer CLs are derived from a Kp,uu decomposition:

```
Kp,uu = (CL_in / CL_out) × (CL_ecf_in / CL_ecf_out) = 0.60
```

The decomposition is **non-unique** — many combinations of four CLs yield the same Kp,uu but different kinetic profiles. The chosen ratios:
- CL_in / CL_out = 0.80 (moderate influx bias, consistent with CNS-active drug)
- CL_ecf_in / CL_ecf_out = 0.75 (moderate ECF entry bias)

#### CL_in (Central → Brain Vascular Influx)

| Property | Value (S) | Value (R) |
|----------|-----------|-----------|
| **New value** | 5.0 L/h | 5.0 L/h |
| **Old value** | 1.2 L/h | 1.2 L/h |
| **Source** | Kp,uu decomposition + Moaddel 2023 |
| **Derivation** | Absolute magnitude set so BBB transfer is rate-limited by perfusion (cerebral blood flow ≈ 3 L/h total) but enhanced by ketamine's high lipophilicity. No enantiomer difference assumed (passive transport dominates). |
| **Uncertainty** | **HIGH** — no direct measurement of individual BBB transfer rates in humans. |

#### CL_out (Brain Vascular → Central Efflux)

| Property | Value (S) | Value (R) |
|----------|-----------|-----------|
| **New value** | 6.25 L/h | 6.25 L/h |
| **Old value** | 1.5 L/h | 1.5 L/h |
| **Source** | Kp,uu decomposition; ratio CL_in/CL_out = 0.80 |
| **Derivation** | CL_out = CL_in / 0.80 = 5.0 / 0.80 = 6.25 L/h. |
| **Uncertainty** | **HIGH** |

#### CL_ecf_in (Brain Vascular → ECF)

| Property | Value (S) | Value (R) |
|----------|-----------|-----------|
| **New value** | 4.0 L/h | 4.0 L/h |
| **Old value** | 0.8 L/h | 0.8 L/h |
| **Source** | Kp,uu decomposition |
| **Uncertainty** | **HIGH** |

#### CL_ecf_out (ECF → Brain Vascular)

| Property | Value (S) | Value (R) |
|----------|-----------|-----------|
| **New value** | 5.33 L/h | 5.33 L/h |
| **Old value** | 1.0 L/h | 1.0 L/h |
| **Source** | Kp,uu decomposition; ratio CL_ecf_in/CL_ecf_out = 0.75 |
| **Derivation** | CL_ecf_out = CL_ecf_in / 0.75 = 4.0 / 0.75 = 5.33 L/h. |
| **Uncertainty** | **HIGH** |

**Kp,uu verification:** (5.0 / 6.25) × (4.0 / 5.33) = 0.80 × 0.75 = **0.60** ✓

### Metabolic Clearances

#### CL_met_NK (Parent → Norketamine N-Demethylation)

| Property | Value (S) | Value (R) |
|----------|-----------|-----------|
| **New value** | 61.6 L/h | 53.8 L/h |
| **Old value** | 18.0 L/h | 14.0 L/h |
| **Source** | Perez-Ruixo 2021 (FRn = 54%) + Hasan 2021 (CL ratio) |
| **Derivation** | CL_met_NK = f_m × CL_total. S-KET: 0.54 × 114 = 61.6 L/h. R-KET: f_m_R ≈ 0.50; CL_total_R = 107.6 L/h (from Hasan S:R ratio); CL_met_NK_R = 0.50 × 107.6 = 53.8 L/h. S:R ratio = 1.15. |
| **Uncertainty** | **MODERATE** — f_m from Perez-Ruixo (N=820). R-KET f_m assumed. |

#### CL_met_HNK (NK → HNK Oxidation)

| Property | Value (S) | Value (R) |
|----------|-----------|-----------|
| **New value** | 4.0 L/h | 4.5 L/h |
| **Old value** | 6.0 L/h | 6.0 L/h |
| **Source** | Weiss & Siegmund 2022 (steady-state cascade) |
| **Derivation** | Backed out from steady-state HNK:KET ratios (46× R, 14× S) in the reduced model. |
| **Uncertainty** | **HIGH** — not directly measured. |

#### CL_other_NK (Other NK Elimination)

| Property | Value (S) | Value (R) |
|----------|-----------|-----------|
| **New value** | 3.27 L/h | 3.68 L/h |
| **Old value** | 4.0 L/h | 4.0 L/h |
| **Source** | Mass balance from f_m_HNK |
| **Derivation** | CL_other = CL_met_HNK × (1/f_m_HNK − 1). S: 4.0 × (1/0.55 − 1) = 3.27 L/h. R: 4.5 × (1/0.55 − 1) = 3.68 L/h. |
| **Uncertainty** | **HIGH** — derivative of CL_met_HNK and f_m_HNK. |

#### CL_out_HNK (Terminal HNK Elimination)

| Property | Value |
|----------|-------|
| **New value** | 0.17 L/h |
| **Old value** | 9.0 L/h |
| **Source** | Weiss & Siegmund 2022 (steady-state HNK:KET ratio) |
| **Derivation** | At steady state: C_HNK/C_NK = f_m_HNK × CL_met_HNK / CL_out_HNK. HNK:NK ≈ 15× → CL_out_HNK = 0.55 × 4.5 / 15 = 0.165 ≈ 0.17 L/h. |
| **Uncertainty** | **VERY HIGH** — most uncertain parameter. The 1-compartment HNK approximation masks multi-compartment kinetics. **FLAG FOR SENSITIVITY ANALYSIS.** |

### Fractions

#### f_m (Fraction of Parent CL → NK)

| Property | Value |
|----------|-------|
| **New value** | 0.54 |
| **Old value** | 0.85 |
| **Source** | Perez-Ruixo 2021 (FRn = 54%) |
| **Derivation** | Directly from Perez-Ruixo population model. |
| **Uncertainty** | **LOW** — directly estimated in N=820. |

#### f_m_HNK (Fraction of NK CL → HNK)

| Property | Value |
|----------|-------|
| **New value** | 0.55 |
| **Old value** | 0.60 |
| **Source** | Weiss & Siegmund 2022 (cascade model) |
| **Uncertainty** | **HIGH** — model-derived, not directly reported. |

### Kp_uu_brain

| Property | Value |
|----------|-------|
| **New value** | 0.6 (unchanged) |
| **Old value** | 0.6 |
| **Source** | Moaddel 2023 (CSF/plasma ratio) + Schmidt & Holzgrabe 2024 (fu) |
| **Derivation** | Moaddel 2023: CSF/plasma total ratio ≈ 0.4. Schmidt: fu ≈ 0.65. Kp,uu = 0.4/0.65 ≈ 0.62, rounded to 0.6. |
| **Uncertainty** | **MODERATE-HIGH** — CSF is a surrogate for brain ECF. |

## CYP2B6 PGx Covariate

The base parameters above represent **CYP2B6\*1/\*1** (wild-type). The PGx module applies genotype-specific scaling factors (θ_2B6) to CL_met_NK:

| Diplotype | θ_2B6 (CL scaling) | Source |
|-----------|-------------------|--------|
| \*1/\*1 | 1.00 (baseline) | Reference |
| \*1/\*6 | ~0.60 | Li 2015: CL 40.6 / 68.1 = 0.60 |
| \*6/\*6 | ~0.32 | Li 2015: CL 21.6 / 68.1 = 0.32 |

**CYP2B6\*6 allele frequency (South Asian):** ~38% → expected \*6/\*6 homozygote frequency ~14–15%.

**Note on Rao 2016:** No significant genotype effect was found at low single oral doses. This suggests the CYP2B6\*6 effect may be **dose/exposure-dependent** (saturable metabolism), which the model should reproduce during validation.

## Comparison Table: Old vs New Parameters

| Parameter | Old (Placeholder) | New (Literature-Derived) | Change | Notes |
|-----------|-------------------|--------------------------|--------|-------|
| V_cen | 30.0 L | 45.0 L | 1.5× | Consistent with Vc/F from popPK |
| V_per | 120.0 L | 707.0 L (S) | 5.9× | Matches Vdss=752 L (Perez-Ruixo) |
| V_vasc | 0.05 L | 0.05 L | — | Unchanged (anatomical) |
| V_ecf | 0.20 L | 0.15 L | 0.75× | Adjusted to anatomical estimate |
| Q_per | 60.0 L/h | 120.0 L/h | 2.0× | Faster distribution for rapid CNS entry |
| CL_in (S/R) | 1.2 / 1.2 | 5.0 / 5.0 | 4.2× | Kp,uu decomposition |
| CL_out (S/R) | 1.5 / 1.5 | 6.25 / 6.25 | 4.2× | Kp,uu decomposition |
| CL_ecf_in (S/R) | 0.8 / 0.8 | 4.0 / 4.0 | 5.0× | Kp,uu decomposition |
| CL_ecf_out (S/R) | 1.0 / 1.0 | 5.33 / 5.33 | 5.3× | Kp,uu decomposition |
| CL_met_NK (S/R) | 18.0 / 14.0 | 61.6 / 53.8 | 3.4× / 3.8× | f_m × CL_total (Perez-Ruixo) |
| CL_met_HNK (S/R) | 6.0 / 6.0 | 4.0 / 4.5 | 0.67× / 0.75× | Weiss & Siegmund cascade |
| CL_other_NK (S/R) | 4.0 / 4.0 | 3.27 / 3.68 | 0.82× / 0.92× | Mass balance from f_m_HNK |
| CL_out_HNK | 9.0 L/h | 0.17 L/h | **0.019×** | Reproduce 46× HNK:KET at SS |
| f_m | 0.85 | 0.54 | 0.64× | Perez-Ruixo FRn = 54% |
| f_m_HNK | 0.60 | 0.55 | 0.92× | Cascade balance |
| Kp_uu_brain | 0.6 | 0.6 | — | Now supported by Moaddel 2023 |

## Parameters Flagged for Sensitivity Analysis

| Priority | Parameter | Uncertainty | Recommended Range |
|----------|-----------|-------------|-------------------|
| **VERY HIGH** | CL_out_HNK | Most uncertain; 1-comp HNK approximation | 0.01–1.0 L/h |
| **HIGH** | Q_per | Not directly reported | 60–200 L/h |
| **HIGH** | CL_in, CL_out, CL_ecf_in, CL_ecf_out | Non-unique Kp,uu decomposition | Test ratio sensitivity |
| **HIGH** | CL_met_HNK | Backed out from SS ratios | 2–8 L/h |
| **HIGH** | CL_other_NK | Derivative of CL_met_HNK, f_m_HNK | 1–6 L/h |
| **HIGH** | f_m_HNK | Model-derived | 0.40–0.70 |

## Consistency Checks

### Predicted vs Published Half-Life

Using t½ ≈ 0.693 × Vdss / CL:

| Species | Vdss (L) | CL (L/h) | Predicted t½ (h) | Published t½ (h) | Discrepancy |
|---------|----------|----------|-------------------|------------------|-------------|
| S-KET | 752 | 114 | 4.58 | 5.2 (Hasan) | −12% |
| R-KET | 638 | 107.6 | 4.11 | 6.1 (Hasan) | −33% |

The simplified formula underestimates t½ in multi-compartment models (terminal phase dominated by V_per redistribution). Full ODE simulation will produce more accurate profiles.

### Metabolic Mass Balance

For S-ketamine:
- Total CL = CL_met_NK + CL_other = 61.6 + 52.4 = **114.0 L/h** ✓
- f_m = CL_met_NK / CL_total = 61.6 / 114.0 = **0.54** ✓

For S-norketamine:
- Total NK CL = CL_met_HNK + CL_other_NK = 4.0 + 3.27 = 7.27 L/h
- f_m_HNK = CL_met_HNK / Total NK CL = 4.0 / 7.27 = **0.55** ✓

### Kp,uu Verification

(CL_in / CL_out) × (CL_ecf_in / CL_ecf_out) = (5.0 / 6.25) × (4.0 / 5.33) = 0.80 × 0.75 = **0.60** ✓

## Notes on Model Reduction

The L1 model reduces the 7–9 compartment popPK models of Kamp 2020, Hasan 2021, and Weiss & Siegmund 2022 to a 4-compartment structure:

1. **HNK kinetics:** Full models track HNK in multiple compartments; L1 uses a single central compartment. CL_out_HNK is an "effective" clearance absorbing distribution effects.
2. **Distribution:** The lumped peripheral compartment (V_per ≈ 700 L) represents all tissue distribution. Q_per is an aggregate of multiple tissue-specific flows.
3. **Enantiomer resolution:** S and R share all distribution parameters (V_cen, V_per, Q_per, BBB CLs) but have enantiomer-specific metabolic clearances. This is reasonable given that fu differences are small and BBB transport is likely passive.
4. **CYP2B6 genotype:** Base parameters represent CYP2B6\*1/\*1. The PGx module applies genotype-specific scaling to CL_met_NK (per Li 2015).
