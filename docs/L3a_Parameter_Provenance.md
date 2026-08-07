# L3a SD Engine Parameter Provenance

**Date:** 2026-05-30
**Module:** L3a spreading depolarization (protective arm)
**Status:** Literature-derived initial values; calibration pending

---

## Parameter Summary

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| K_thr_0 | 12.0 | mM K+ | Somjen 2001; SD threshold |
| K_rest | 3.0 | mM K+ | Physiological resting |
| K_peak | 60.0 | mM K+ | SD peak extracellular K+ |
| D_SD_0 | 1.0 | min | SD depolarization duration |
| τ_rec_0 | 5.0 | min | Recovery time constant |
| λ_SD_0 | 2.0 | events/h | Baseline SD rate in injury |
| κ | 5.0 | — | B_pyr threshold gain |
| κ_D | 2.0 | — | B_pyr duration gain |
| κ_rec | 1.0 | — | B_pyr recovery gain |
| D_K | 2.5 | mm²/min | K+ diffusion coefficient |

---

## Key Literature Anchors

### SD Suppression Threshold (Carlson 2018/2019)

| Property | Value |
|----------|-------|
| **Source** | Carlson 2018/2019; Gregers 2020 systematic review |
| **Finding** | Ketamine > 1.15 mg/kg/h suppresses SD (OR 13.84, 95% CI 1.99-1000) |
| **Use** | Qualification test: model must reproduce this threshold |

### SD Electrophysiology

| Property | Value |
|----------|-------|
| **K+ threshold** | ~12 mM for SD initiation (Somjen 2001) |
| **K+ peak** | ~60 mM during SD event |
| **Propagation speed** | 2-8 mm/min |
| **Duration** | ~1 min depolarization, ~5 min recovery |

### B_pyr Coupling Rationale

| Mechanism | Effect | Gain |
|-----------|--------|------|
| Threshold elevation | K_thr = K_thr_0 * (1 + κ·B_pyr) | κ = 5.0 |
| Duration shortening | D_SD = D_SD_0 / (1 + κ_D·B_pyr) | κ_D = 2.0 |
| Recovery acceleration | τ_rec = τ_rec_0 / (1 + κ_rec·B_pyr) | κ_rec = 1.0 |

---

## References

1. Carlson AP, et al. Neurocrit Care 2018; 29:473-482
2. Sánchez-Porras R, et al. J Neurotrauma 2022
3. Somjen GG. Physiol Rev 2001; 81:1065-1096
4. Hübel N, Dahlem MA. PLoS Comput Biol 2014; 10:e1003773
5. Dreier JP, et al. Ann Neurol 2011; 70:549-560
6. Hartings JA, et al. Brain 2017; 140:2562-2575
7. Gregers MCT, et al. Neurocrit Care 2020; 33:273-282
