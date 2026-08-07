# L3b NRHypo Engine Parameter Provenance

**Date:** 2026-05-30
**Module:** L3b NRHypo toxic arm (interneuron disinhibition)
**Status:** Literature-derived initial values; **strongly informative priors required**

---

## Parameter Summary

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| I_GABA_0 | 0.4 | — | Olney & Farber 1995; ~40% of pyramidal activity GABA-inhibited |
| B_int_thresh | 0.3 | — | Morgan 2014; interneuron 10× sensitivity → threshold at low B_int |
| g_gain | 10.0 | h⁻¹ | Olney 1989; dose-dependent injury scaling |
| g_exponent | 2.0 | — | Convex injury (quadratic); supralinear above threshold |
| tau_Glu | 0.5 | h | Glutamate clearance; metabolic dependence |
| Glu_0 | 2.0 | µM | Resting extracellular glutamate |
| Glu_max | 100.0 | µM | Maximum pathological glutamate (ischemia/SD levels) |

---

## Detailed Derivation

### I_GABA_0 (Baseline Inhibitory Drive)

| Property | Value |
|----------|-------|
| **Source** | Olney & Farber 1995; Yan & Rein 2021 |
| **Derivation** | ~40% of pyramidal neuron activity is under GABAergic inhibition. NMDAR block on interneurons removes this inhibition → pyramidal disinhibition. |
| **Uncertainty** | **MODERATE** — varies by brain region (PCC/RSC more sensitive) |

### B_int_thresh (Disinhibition Threshold)

| Property | Value |
|----------|-------|
| **Source** | Morgan et al. 2014; Li et al. 2002 |
| **Derivation** | GABA interneurons are ~10× more sensitive to NMDAR antagonists than pyramidal neurons. At low doses (B_int ≈ 0.3), disinhibition begins to manifest. Below this, compensatory mechanisms maintain inhibition. |
| **Uncertainty** | **HIGH** — dose-response data is sparse in humans |

### g_gain and g_exponent (Convex Injury Function)

| Property | Value |
|----------|-------|
| **Source** | Olney et al. 1989, 1991; Fix et al. 1995 |
| **Derivation** | Low doses → mitochondrial dilation (reversible). Higher doses → neuronal death (irreversible). The transition is supralinear → convex gain function with exponent ≥ 2. |
| **Uncertainty** | **VERY HIGH** — most uncertain parameter group; load-bearing for upper window boundary |

### tau_Glu (Glutamate Clearance)

| Property | Value |
|----------|-------|
| **Source** | Metabolic dependence; SD literature |
| **Derivation** | Glutamate clearance depends on astrocyte uptake and metabolic supply. In ischemia, clearance is impaired → longer τ. Baseline: ~30 min. |
| **Uncertainty** | **MODERATE** — depends on tissue metabolic state |

---

## Mechanism Summary

```
B_int (interneuron NMDAR block)
    ↓
I_GABA reduced (disinhibition)
    ↓
E_pyr increased (pyramidal disinhibition)
    ↓
Glutamate surge (AMPA/kainate activation)
    ↓
T_NRHypo = ∫ g(B_int(τ)) · (Glu(τ) - Glu_0)/Glu_max dτ
    ↓
NRHypo injury (cumulative, irreversible)
```

---

## References

1. Olney JW, Farber NB. Arch Gen Psychiatry 1995; 52:998-1003
2. Olney JW, Labruyere J, Price MT. Science 1989; 244:1360-1362
3. Li Q, Clark S, Lewis DV, Wilson WA. J Neurosci 2002; 22:3070-3080
4. Morgan CJ, et al. Front Psychiatry 2014; 5:149
5. Yan Z, Rein B. Mol Psychiatry 2021; 27:445-465
6. Fix AS, et al. Neurotoxicology 1995; 16:417-424
7. Corso TD, et al. Exp Neurol 1997; 143:276-284
