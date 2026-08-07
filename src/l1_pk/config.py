"""Centralized typed configuration schema for the L1 PBPK/PK model.

Single source of truth for all L1 parameters (TechSpec §2-§4). Every
numeric field carries an explicit unit in its declaration so the ODE engine
and the verification suite consume one canonical, dimensionally-checked
representation.

Unit conventions (consistent throughout L1):
  - amount        : mg
  - concentration : mg/L
  - volume        : L
  - flow / CL     : L/h
  - time          : h
  - dose rate     : mg/h
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical enantiomer labels used as dictionary keys throughout L1.
ENANTIOMERS = ("S", "R")

# Registry of every parameter's physical unit (TechSpec §2, §3). Consumed by
# the verification suite's dimensional-consistency check.
UNIT_REGISTRY: dict[str, str] = {
    "V_cen": "L",
    "V_per": "L",
    "V_vasc": "L",
    "V_ecf": "L",
    "Q_per": "L/h",
    "CL_in": "L/h",
    "CL_out": "L/h",
    "CL_ecf_in": "L/h",
    "CL_ecf_out": "L/h",
    "CL_met_NK": "L/h",
    "CL_other_parent": "L/h",
    "CL_met_HNK": "L/h",
    "CL_other_NK": "L/h",
    "CL_out_HNK": "L/h",
    "f_m": "dimensionless",
    "f_m_HNK": "dimensionless",
    "Kp_uu_brain": "dimensionless",
}


@dataclass(frozen=True)
class CompartmentVolumes:
    """Compartment volumes V_i [L] (TechSpec §2).

    Central and peripheral volumes are ESTIMATED by scripts/run_l1_calibration.py
    from digitized human intravenous concentration-time data; 95% bootstrap
    intervals are in docs/L1_Calibration.md and results/l1_calibration.json.
    They are no longer taken from the intranasal esketamine population analysis,
    whose reported volume is an apparent value inflated by bioavailability.
    """

    V_cen: float = 49.72   # calibrated [45.5 to 57.3]
    V_per: float = 234.37  # calibrated [220 to 316]; Vss = V_cen + V_per = 284 L
    V_vasc: float = 0.05   # brain vascular sub-compartment (~50 mL), anatomical
    V_ecf: float = 0.15    # brain ECF (PD driver; ~150 mL), anatomical


@dataclass(frozen=True)
class Flows:
    """Inter-compartmental flow Q_ij [L/h] (TechSpec §3.1)."""

    Q_per: float = 75.54  # calibrated [56.5 to 86.1]


@dataclass(frozen=True)
class Clearances:
    """Clearances [L/h], enantiomer-resolved where applicable (TechSpec §3-§4).

    Per-enantiomer fields map enantiomer label ("S"/"R") -> value to capture
    enantioselective clearance (TechSpec §3.2). genotype scaling theta_2B6(g)
    is applied externally by the PGx module onto ``CL_met_NK``.

    Parent and norketamine clearances are ESTIMATED by
    scripts/run_l1_calibration.py from digitized human intravenous
    concentration-time data (Kamp 2020), with weakly informative penalties from
    the Hasan 2021 intravenous arm. Blood-brain transfer clearances are FIXED:
    no human dataset identifies them, and their ratios set Kp,uu = 0.6
    (Moaddel 2023). Hydroxynorketamine parameters are FIXED at values
    constrained by Weiss & Siegmund 2022; hydroxynorketamine was excluded from
    the calibration and feeds no downstream layer. See docs/L1_Calibration.md.
    """

    # BBB transfer central <-> brain_vasc. FIXED, not identified by any data.
    # Kp,uu decomposition: (CL_in/CL_out)×(CL_ecf_in/CL_ecf_out) = 0.60
    CL_in: dict[str, float] = field(default_factory=lambda: {"S": 5.0, "R": 5.0})
    CL_out: dict[str, float] = field(default_factory=lambda: {"S": 6.25, "R": 6.25})
    # BBB transfer brain_vasc <-> ecf. FIXED.
    CL_ecf_in: dict[str, float] = field(default_factory=lambda: {"S": 4.0, "R": 4.0})
    CL_ecf_out: dict[str, float] = field(default_factory=lambda: {"S": 5.33, "R": 5.33})
    # Total parent clearance, routed entirely through the N-demethylation node;
    # the fraction f_m of it forms norketamine (see MetaboliteFractions). This
    # parameterisation keeps parent elimination and metabolite formation
    # non-redundant. CALIBRATED: S 92.65 [88.4 to 95.7]; R follows the published
    # S:R clearance ratio of 1.059 (Hasan 2021 intravenous arm).
    CL_met_NK: dict[str, float] = field(default_factory=lambda: {"S": 92.65, "R": 87.51})
    # Absorbed into CL_met_NK by the parameterisation above.
    CL_other_parent: dict[str, float] = field(default_factory=lambda: {"S": 0.0, "R": 0.0})
    # Total norketamine clearance; the fraction f_m_HNK of it forms HNK.
    # CALIBRATED: 6.836 [5.11 to 7.33]
    CL_met_HNK: dict[str, float] = field(default_factory=lambda: {"S": 6.836, "R": 6.836})
    # Absorbed into CL_met_HNK by the parameterisation above.
    CL_other_NK: dict[str, float] = field(default_factory=lambda: {"S": 0.0, "R": 0.0})
    # Terminal HNK elimination. FIXED to the Weiss & Siegmund steady-state ratio.
    CL_out_HNK: float = 2.0


@dataclass(frozen=True)
class MetaboliteFractions:
    """Dimensionless metabolic routing fractions (TechSpec §3.2)."""

    # CALIBRATED [0.512 to 0.545], estimated freely within 0.40 to 0.95. That it
    # lands near the independently reported metabolic fraction is a check, not an
    # input: no prior pinned it there.
    f_m: float = 0.5228
    f_m_HNK: float = 0.55  # FIXED; Weiss & Siegmund cascade


@dataclass(frozen=True)
class L1Parameters:
    """Aggregate L1 parameter set — the single source of truth."""

    volumes: CompartmentVolumes = field(default_factory=CompartmentVolumes)
    flows: Flows = field(default_factory=Flows)
    clearances: Clearances = field(default_factory=Clearances)
    fractions: MetaboliteFractions = field(default_factory=MetaboliteFractions)
    # Reference partition coefficient (pseudo-steady-state target, TechSpec §3.3).
    Kp_uu_brain: float = 0.6

    def with_zero_clearance(self) -> "L1Parameters":
        """Return a copy with all elimination/metabolism clearances zeroed.

        Distribution and BBB transfer (reversible) are retained so the system
        becomes closed; used by the mass-conservation verification test
        (TechSpec §7, Protocol §10).
        """
        cl = self.clearances
        closed = Clearances(
            CL_in=dict(cl.CL_in),
            CL_out=dict(cl.CL_out),
            CL_ecf_in=dict(cl.CL_ecf_in),
            CL_ecf_out=dict(cl.CL_ecf_out),
            CL_met_NK={e: 0.0 for e in ENANTIOMERS},
            CL_other_parent={e: 0.0 for e in ENANTIOMERS},
            CL_met_HNK={e: 0.0 for e in ENANTIOMERS},
            CL_other_NK={e: 0.0 for e in ENANTIOMERS},
            CL_out_HNK=0.0,
        )
        return L1Parameters(
            volumes=self.volumes,
            flows=self.flows,
            clearances=closed,
            fractions=self.fractions,
            Kp_uu_brain=self.Kp_uu_brain,
        )


def default_parameters() -> L1Parameters:
    """Return the default L1 parameter set.

    Values are literature-derived from published popPK models (Perez-Ruixo
    2021, Hasan 2021, Kamp 2020, Weiss & Siegmund 2022, Moaddel 2023).
    See docs/L1_Parameter_Provenance.md for full derivation and source
    mapping. Final WP1 calibration will replace these with Bayesian NLME
    posteriors (Calibration Strategy Step 1).
    """
    return L1Parameters()
