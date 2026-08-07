# Identifiability of the injury layer

Produced by `scripts/run_identifiability.py`; raw output in
`results/identifiability.json`.

---

## 1. The problem

Six coefficients of the protective and toxic layers cannot be estimated from any
available outcome data:

| Symbol | Meaning | Nominal |
|---|---|---|
| alpha | excitotoxic injury coefficient | 1.0 |
| beta | protective coefficient | 0.8 |
| gamma | hypofunction injury weight | 50 |
| g_gain | hypofunction gain | 50 |
| theta | interneuron toxic threshold | 0.3 |
| kappa | spreading depolarisation threshold gain | 5.0 |

An earlier version assigned illustrative values and sampled 300 parameter sets
over plausible ranges. That answered a weaker question than the one that
matters: it reported how often a conclusion held across the values sampled, not
whether the model can distinguish those values at all.

## 2. The injury layer is linear in its coefficients

Once the layer 1 and layer 2 trajectories are fixed, and neither depends on any
of the six, the injury equation is linear in `alpha`, `beta`, and the product
`gamma * g_gain`. Terminal injury at dose D is exactly

```
I(D) = alpha * P(D; kappa)  -  beta * Q(D; kappa)  +  Gamma * R(D; theta)
```

with `Gamma = gamma * g_gain` and the three basis integrals

```
P = INT Phi_exc dt
Q = INT B_pyr * Phi_exc dt
R = INT max(0, B_int - theta)^n dt
```

which depend on dose, on `kappa`, and on `theta`, but on no coefficient.

Verified numerically, not merely derived: for twelve random coefficient vectors,
the closed form and full integration of the differential equation agree to a
maximum relative error of 3e-5, that is to solver tolerance.

## 3. Consequences

**gamma and g_gain are confounded.** They enter only through their product. No
measurement of the injury readout, however precise, separates them. This is why
the earlier variance-based sensitivity analysis, which varied `g_gain`, could
omit `gamma`: varying one already spanned the other. The current analysis
includes both and reports the confounding, and the empirical signature is
visible in the Sobol output, where the two have almost identical total-order
indices (0.20 and 0.23) and their pairwise second-order index is the second
largest of any pair.

**alpha sets scale, not shape.** Dividing through leaves the curve depending
only on `beta/alpha`, `Gamma/alpha`, `theta`, and `kappa`.

**Six free coefficients reduce to four shape-determining quantities.**

**The coefficient space can be swept, not sampled.** Because P, Q, and R do not
depend on the coefficients, they are computed once per dose and per (kappa,
theta) and every combination is then evaluated at negligible cost.

## 4. What the window requires

Exhaustive sweep, 932 263 combinations of (kappa, theta, beta/alpha,
Gamma/alpha):

| Result | Value |
|---|---|
| Window present above the critical toxic weight | 100.00% of 569 667 sets |
| Window present below it | 0% |
| Cases where raising Gamma removed an existing window | 0 |
| Bolus worse than the dose-matched infusion | 100% of all 932 263 sets |
| Critical Gamma/alpha, median (range) | 57 (21 to 134) |
| Nominal Gamma/alpha | 2500 |
| Margin | 53-fold; above critical for 100% of (kappa, theta, beta/alpha) triples |
| Equivalent critical gamma at nominal g_gain | 0.95, against a nominal 50 |

The transition is a **boundary, not a gradual fade**. This makes the earlier
framing ("a window appeared in 299 of 300 parameterisations") less informative
than it looked: 299 out of 300 was a property of the box that happened to be
sampled, not of the model.

The bolus-versus-infusion ordering is the more robust result of the two. It held
in **every** coefficient set examined, including those in which no window exists
at all, so it does not depend on the window existing.

## 5. Why the condition is not arbitrary

`Gamma` exceeding its critical value is the formal statement that
N-methyl-D-aspartate receptor antagonists are neurotoxic at sufficient exposure.
That proposition is not in serious doubt: it is the Olney lesion, reproduced
across species and compounds for more than three decades. The window in this
model is therefore a quantified consequence of an independently established fact
about the drug class, not a free parameter tuned until a window appeared.

What the value of `Gamma` determines is **where** the upper limit falls, not
**whether** there is one. That is the quantity a prospective study would measure
(see `scripts/run_prospective_design.py`).

## 6. What this does not establish

The decomposition says which coefficient combinations the model can distinguish
**given its structure**. It does not validate the structure. If the true relation
between interneuron blockade and injury is not convex above a threshold, or if
the protective term should not scale with the excitotoxic flux, no analysis of
the coefficients will help.

Nor does it make the injury readout dimensional. `I` is in arbitrary units
before and after. The conclusions that survive the sweep are ordinal ones, about
the shape of a curve and the ranking of regimens, and those are the only
conclusions drawn.

## 7. Reproducing

```bash
PYTHONPATH=. python scripts/run_identifiability.py   # ~5 min
```
