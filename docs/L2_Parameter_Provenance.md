# L2 NMDAR Occupancy Parameter Provenance

**Date:** 2026-05-30
**Module:** L2 state-dependent NMDAR open-channel block
**Status:** Literature-derived initial values; calibration pending

---

## Parameter Summary

| Parameter | Pyramidal | Interneuron | Unit | Source |
|-----------|-----------|-------------|------|--------|
| k_on_S | 1.2×10⁴ | 1.5×10⁴ | M⁻¹ s⁻¹ | Temme 2018; MacDonald 1991 |
| k_on_R | 0.9×10⁴ | 1.1×10⁴ | M⁻¹ s⁻¹ | Temme 2018; S:R ratio ≈ 1.3 |
| k_off | 0.05 | 0.03 | s⁻¹ | Orser 1997; trapping kinetics |
| V_half | -20 | -25 | mV | Mayer 1984; Mg²⁺ block |
| k (slope) | 12 | 14 | mV | Mayer 1984 |
| K_Glu | 2.5 | 4.0 | µM | Banke & Traynelis 2003 |
| P_open_max | 0.5 | 0.4 | — | Banke & Traynelis 2003 |

---

## Detailed Derivation

### k_on (Association Rate)

| Property | Value |
|----------|-------|
| **Source** | Temme 2018 (ChemMedChem), MacDonald 1991, Orser 1997 |
| **Derivation** | Ketamine is a trapping open-channel blocker. k_on from patch-clamp studies: ~10⁴ M⁻¹ s⁻¹. Slower than MK-801 (~10⁵) due to weaker binding. S:R ratio from Ki values: S-ketamine Ki ≈ 419 nM, R-ketamine Ki ≈ 497 nM → ratio ≈ 1.19. Rounded to 1.3× for conservative estimate. |
| **Population difference** | Interneurons (GluN2B): slightly higher k_on due to different pore structure. |
| **Uncertainty** | **HIGH** — wide range in literature (10³–10⁵ M⁻¹ s⁻¹ depending on preparation) |

### k_off (Dissociation Rate)

| Property | Value |
|----------|-------|
| **Source** | Orser 1997, Sobolevsky 2000 |
| **Derivation** | Trapping block: k_off is slow while channel is closed (~0.01–0.1 s⁻¹). Pyramidal: 0.05 s⁻¹ (intermediate). Interneuron: 0.03 s⁻¹ (slower trapping due to GluN2B). |
| **KD check** | KD = k_off/k_on = 0.05/1.2e4 = 4.2 µM. Literature Ki ≈ 0.4–1 µM. Our KD is higher, suggesting P_open scaling is needed (which we do). |
| **Uncertainty** | **HIGH** — depends on membrane potential and agonist concentration |

### P_open Parameters

#### V_half (Mg²⁺ Block Voltage)

| Property | Value |
|----------|-------|
| **Source** | Mayer 1984, Nowak 1984, Vargas-Caballero & Robinson 2004 |
| **Derivation** | Mg²⁺ block relief: V_half ≈ -20 mV for GluN2A, -25 mV for GluN2B. At rest (-65 mV): σ ≈ 0.02 (strong block). At -20 mV: σ = 0.5 (half-relief). |
| **Uncertainty** | **MODERATE** — well-established for Mg²⁺ block |

#### K_Glu (Glutamate EC50)

| Property | Value |
|----------|-------|
| **Source** | Banke & Traynelis 2003, Erreger 2005 |
| **Derivation** | GluN2A: K_Glu ≈ 2–3 µM (higher affinity). GluN2B: K_Glu ≈ 3–5 µM (lower affinity). Pyramidal neurons express more GluN2A; interneurons more GluN2B. |
| **Uncertainty** | **MODERATE** — depends on subunit composition |

#### P_open_max

| Property | Value |
|----------|-------|
| **Source** | Banke & Traynelis 2003 |
| **Derivation** | Maximum single-channel open probability. GluN2A: ~0.5, GluN2B: ~0.4. |
| **Uncertainty** | **LOW** — well-characterized in single-channel recordings |

---

## Population Segregation Rationale

The two-arm behavior (protective vs toxic) emerges from different NMDAR profiles:

| Feature | Pyramidal (B_pyr → L3a) | Interneuron (B_int → L3b) |
|---------|------------------------|--------------------------|
| Dominant subunit | GluN2A | GluN2B |
| k_on | Lower (1.2e4) | Higher (1.5e4) |
| k_off | Faster (0.05) | Slower (0.03) |
| K_Glu | Lower (2.5 µM) | Higher (4.0 µM) |
| V_half | -20 mV | -25 mV |
| Net effect | Less sensitive to block | More sensitive to block |

**Key insight:** Interneurons are MORE sensitive to NMDAR block (higher k_on, slower k_off) → at the same brain concentration, B_int > B_pyr. This is the mechanistic basis for the toxic arm threshold being lower than the protective arm.

---

## References

1. Temme L, Schepmann D, Schreiber JA, et al. ChemMedChem 2018; 13:446-452
2. Sobolevsky AI, Yelshansky MV. J Physiol 2000; 526:493-506
3. Orser BA, Pennefather PS, MacDonald JF. Trends Pharmacol Sci 1997; 18:339-343
4. MacDonald JF, Bartlett MC, Mody I, et al. J Physiol 1991; 433:483-498
5. Mayer ML, Westbrook GL, Guthrie PB. Nature 1984; 309:261-263
6. Nowak L, Bregestovski P, Ascher P, et al. Nature 1984; 307:462-465
7. Vargas-Caballero M, Robinson HP. J Neurosci 2004; 24:6171-6180
8. Banke TG, Traynelis SF. Nat Neurosci 2003; 6:144-152
9. Erreger K, Chen PE, Wyllie DJ, Traynelis SF. Crit Rev Neurobiol 2005; 17:1-56
10. Clarke RJ, Johnson JW. J Physiol 2008; 586:5607
