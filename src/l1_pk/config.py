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
    """Compartment volumes V_i [L] (TechSpec §2)."""

    V_cen: float = 45.0   # central/plasma; Perez-Ruixo Vc inferred from Vc/F
    V_per: float = 650.0  # peripheral; Kamp V2=115, adjusted for Hasan Vdss range
    V_vasc: float = 0.05  # brain vascular sub-compartment (~50 mL)
    V_ecf: float = 0.15   # brain ECF (PD driver; ~150 mL)


@dataclass(frozen=True)
class Flows:
    """Inter-compartmental flow Q_ij [L/h] (TechSpec §3.1)."""

    Q_per: float = 250.0  # central <-> peripheral; Kamp Q=126, tuned for t½≈5-6h


@dataclass(frozen=True)
class Clearances:
    """Clearances [L/h], enantiomer-resolved where applicable (TechSpec §3-§4).

    Per-enantiomer fields map enantiomer label ("S"/"R") -> value to capture
    enantioselective clearance (TechSpec §3.2). genotype scaling theta_2B6(g)
    is applied externally by the PGx module onto ``CL_met_NK``.

    Literature-derived values (May 2026): Perez-Ruixo 2021 (CL, Vss, f_m),
    Hasan 2021 (enantiomer-specific Vdss/CL), Weiss & Siegmund 2022 (HNK
    cascade), Moaddel 2023 (Kp,uu). See docs/L1_Parameter_Provenance.md.
    """

    # BBB transfer central <-> brain_vasc.
    # Kp,uu decomposition: (CL_in/CL_out)×(CL_ecf_in/CL_ecf_out) = 0.60
    CL_in: dict[str, float] = field(default_factory=lambda: {"S": 5.0, "R": 5.0})
    CL_out: dict[str, float] = field(default_factory=lambda: {"S": 6.25, "R": 6.25})
    # BBB transfer brain_vasc <-> ecf.
    CL_ecf_in: dict[str, float] = field(default_factory=lambda: {"S": 4.0, "R": 4.0})
    CL_ecf_out: dict[str, float] = field(default_factory=lambda: {"S": 5.33, "R": 5.33})
    # N-demethylation parent -> NK, enantioselective.
    # f_m_S=0.54 × CL_S=114 = 61.6; f_m_R=0.50 × CL_R=107.6 = 53.8
    CL_met_NK: dict[str, float] = field(default_factory=lambda: {"S": 61.6, "R": 53.8})
    # Other (non-NK) parent elimination; CL_total = CL_met_NK / f_m
    # CL_other_parent_S = 114.0 - 61.6 = 52.4; CL_other_parent_R = 107.6 - 53.8 = 53.8
    CL_other_parent: dict[str, float] = field(default_factory=lambda: {"S": 52.4, "R": 53.8})
    # NK -> HNK oxidation, enantioselective.
    CL_met_HNK: dict[str, float] = field(default_factory=lambda: {"S": 4.0, "R": 4.5})
    # Other (non-HNK) NK elimination.
    CL_other_NK: dict[str, float] = field(default_factory=lambda: {"S": 3.27, "R": 3.68})
    # Terminal HNK elimination; Kamp HNK CL=76.2 L/h (7-comp model);
    # reduced to 2.0 L/h for 1-comp HNK model to match Weiss & Siegmund SS ratio.
    CL_out_HNK: float = 2.0


@dataclass(frozen=True)
class MetaboliteFractions:
    """Dimensionless metabolic routing fractions (TechSpec §3.2)."""

    f_m: float = 0.54      # fraction of parent CL -> NK; Perez-Ruixo FRn=54%
    f_m_HNK: float = 0.55  # fraction of NK CL -> HNK; Weiss & Siegmund cascade


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
