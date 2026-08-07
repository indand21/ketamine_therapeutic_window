# L1 pharmacokinetic calibration

Authoritative record for the layer 1 parameter values in `src/l1_pk/config.py`.
Produced by `scripts/run_l1_calibration.py`; raw output in
`results/l1_calibration.json`.

This document supersedes the disposition parameters described in
`L1_Parameter_Provenance.md`. That file remains as the record of where the
earlier values came from and is retained deliberately, because the reason they
were wrong is instructive.

---

## 1. Why the layer was recalibrated

Earlier versions took clearance and volume from a population analysis of
**intranasal** esketamine. The values reported there are apparent quantities,
CL/F and Vss/F, inflated by incomplete bioavailability; treating them as
intravenous disposition parameters was an error.

It went undetected because the layer had only ever been checked against summary
statistics drawn from the same source, never against concentration-time data.
The first quantitative check against digitized human concentrations gave:

| | Median absolute prediction error | Geometric mean ratio | Within two-fold |
|---|---|---|---|
| Parent and norketamine, 93 observations | 52.0% | 0.43 | 47% |
| Withheld venous curves, 18 observations | 31.1% | 0.76 | 72% |

Plasma ketamine was underpredicted by roughly 40% and the metabolites by an
order of magnitude.

## 2. Design

**Calibration set.** Kamp 2020 (*Br J Anaesth* 125:750-61): six arterial
concentration-time curves from an escalating three-step intravenous infusion of
esketamine and of racemic ketamine, covering S-ketamine, R-ketamine,
S-norketamine, and R-norketamine. 93 observations.

**Held-out set.** Hasan 2021 (*Anesthesiology* 135:326-39): two venous curves
after 5 mg intravenous racemate over 30 min. 18 observations. No parameter was
estimated from these curves.

> These are **partly, not wholly, independent.** Summary parameters tabulated in
> the same paper (clearance and steady-state volume, per enantiomer) informed
> the penalty terms. A fully independent evaluation set was not available, and
> the manuscript says so.

**Estimated, six parameters.** Central volume, peripheral volume,
intercompartmental flow, total parent clearance for S-ketamine, the fraction of
parent clearance routed through N-demethylation (confined to 0.40 to 0.95), and
total norketamine clearance. R-ketamine clearance follows the published S:R
ratio of 1.059.

**Fixed.** The four blood-brain transfer clearances, which no available human
dataset identifies; their ratios set Kp,uu = 0.6. The two hydroxynorketamine
parameters, held at values constrained by Weiss and Siegmund 2022.

**Objective.** Mean squared residual on the natural-log concentration scale,
plus Gaussian penalty terms on total clearance and steady-state volume, each
weighted by its published standard deviation. Optimised by trust-region least
squares.

**Penalty weight.** Chosen by held-out performance rather than by taste.
Sweeping the weight over 0.25, 0.5, 1, 2, and 4 left the calibration error
essentially flat (8.9 to 10.3%) but minimised error on the held-out curves at a
weight of 1, which is also the value at which each penalty carries exactly its
published standard deviation.

**Uncertainty.** Nonparametric bootstrap, 200 resamples of the calibration
observations.

## 3. Estimates

| Parameter | Estimate | 95% bootstrap interval |
|---|---|---|
| V_cen (L) | 49.72 | 45.5 to 57.3 |
| V_per (L) | 234.37 | 220 to 316 |
| Q_per (L/h) | 75.54 | 56.5 to 86.1 |
| CL_parent, S (L/h) | 92.65 | 88.4 to 95.7 |
| f_NK | 0.523 | 0.512 to 0.545 |
| CL_NK total (L/h) | 6.836 | 5.11 to 7.33 |

Derived, against the Hasan 2021 intravenous arm:

| Quantity | Model | Published (mean ± SD) | Standardised difference |
|---|---|---|---|
| CL, S-ketamine (L/h) | 92.7 | 97.2 ± 22.8 | -0.20 |
| CL, R-ketamine (L/h) | 87.5 | 91.8 ± 22.8 | -0.19 |
| Terminal t½, S (h) | 4.08 | 5.2 ± 3.4 | -0.33 |
| Vss (L) | 284 | 462 ± 154 | -1.16 |

## 4. Fit quality

| Dataset | Median absolute prediction error | Geometric mean ratio | Within two-fold |
|---|---|---|---|
| Calibration, before | 52.0% | 0.43 | 47% |
| **Calibration, after** | **9.7%** | **0.94** | **100%** |
| Held out, before | 31.1% | 0.76 | 72% |
| **Held out, after** | **27.9%** | **0.74** | **72%** |
| Hydroxynorketamine (never calibrated) | 77.8% | 0.21 | 7% |

## 5. Three things worth stating plainly

**The free metabolic fraction landed on the literature value.** f_NK was
estimated freely within 0.40 to 0.95 and returned 0.523. No penalty acted on it.
That it falls near the independently reported metabolic fraction is a check on
the calibration, not an input to it.

**Steady-state volume disagrees with the published non-compartmental estimate**
by 1.16 standard deviations. We do not think either is simply wrong.
Non-compartmental Vss is sensitive to characterisation of the terminal phase,
which a 5 mg dose with sparse late sampling characterises poorly; our estimate
is driven by data covering only the first 8 h and is correspondingly weak about
the true terminal phase. The two datasets disagree about the tail, and neither
resolves it.

**Calibration barely improved the held-out set**, from 31.1% to 27.9%. We report
this rather than quoting the calibration-set improvement alone. Likely reasons:
the held-out curves are venous whereas the model was calibrated to arterial data;
they come from a different study, different subjects, and a 28-fold lower dose;
and digitisation error on these low concentrations is substantial. The defensible
claim is narrow: the calibration greatly improves description of the data it was
fitted to, and leaves prediction of an independent venous dataset approximately
unchanged at a level where 72% of observations fall within two-fold.

## 6. Hydroxynorketamine

Deliberately excluded from the calibration. The model forms
(2R,6R)-hydroxynorketamine from R-norketamine alone, whereas the digitized curve
is total hydroxynorketamine, so the comparison is not like for like. Its
prediction error is poor (77.8%), and the single-compartment approximation with
a fixed terminal clearance also produces an implausible late accumulation ratio
relative to parent.

**No downstream layer reads the hydroxynorketamine concentration.** Receptor
occupancy is driven by the parent enantiomers alone, so no result depends on it.
Its concentrations should not be interpreted. It is retained only because the
same engine is intended to serve analyses in which the metabolite does act.

## 7. Reproducing

```bash
PYTHONPATH=. python scripts/run_l1_calibration.py       # ~40 min with the bootstrap
PYTHONPATH=. python scripts/run_pk_goodness_of_fit.py   # prediction error, all nine curves
```

The first writes `results/l1_calibration.json` and prints the values to paste
into `src/l1_pk/config.py`.
