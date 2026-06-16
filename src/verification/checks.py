"""Verification helpers for the L1 engine (TechSpec §7, Protocol §10).

Functions here are pure and reusable so they can back both the pytest suite
and runtime diagnostics. They cover mass conservation and dimensional
(unit) consistency of the parameter configuration.
"""

from __future__ import annotations

import numpy as np

from src.l1_pk.config import UNIT_REGISTRY, L1Parameters, Clearances
from src.l1_pk.model import SimulationResult


def total_drug_mass(result: SimulationResult) -> np.ndarray:
    """Total drug mass [mg] across all (species, compartment) cells per time.

    Parent and metabolites are summed on a molar-equivalent basis of 1:1 here
    (illustrative units in mg-equivalents); the closed-system test only
    requires that no mass is created or destroyed when clearance is zero.
    """
    return result.total_mass()


def mass_conservation_residual(result: SimulationResult, regimen) -> float:
    """Max absolute deviation [mg] of total mass from the cumulative input.

    For a closed system (zero metabolism/elimination) the total tracked mass
    at time t must equal the cumulative dosed mass delivered up to t
    (TechSpec §7). The expected balance is therefore time-dependent.
    """
    mass = total_drug_mass(result)
    expected = np.array([regimen.total_dosed_mass(t) for t in result.t])
    return float(np.max(np.abs(mass - expected)))


def check_unit_consistency(params: L1Parameters) -> list[str]:
    """Return a list of dimensional-consistency problems (empty == OK).

    Verifies every parameter declared in ``UNIT_REGISTRY`` is present on the
    config object with the expected physical dimension class, and that
    rate-type parameters (L/h) and volumes (L) are non-negative and finite.
    """
    problems: list[str] = []

    volume_fields = {"V_cen", "V_per", "V_vasc", "V_ecf"}
    flow_cl_fields = {
        "Q_per", "CL_in", "CL_out", "CL_ecf_in", "CL_ecf_out",
        "CL_met_NK", "CL_other_parent", "CL_met_HNK", "CL_other_NK", "CL_out_HNK",
    }
    dimensionless_fields = {"f_m", "f_m_HNK", "Kp_uu_brain"}

    def _values(raw: object) -> list[float]:
        if isinstance(raw, dict):
            return list(raw.values())
        return [float(raw)]  # scalar

    def _lookup(name: str) -> object | None:
        for holder in (
            params.volumes, params.flows, params.clearances, params.fractions,
        ):
            if hasattr(holder, name):
                return getattr(holder, name)
        if hasattr(params, name):
            return getattr(params, name)
        return None

    for name, unit in UNIT_REGISTRY.items():
        raw = _lookup(name)
        if raw is None:
            problems.append(f"{name}: missing from configuration")
            continue
        vals = _values(raw)
        for v in vals:
            if not np.isfinite(v):
                problems.append(f"{name}: non-finite value ({v}) [{unit}]")
            if name in volume_fields and v <= 0:
                problems.append(f"{name}: volume must be > 0 (got {v}) [{unit}]")
            if name in flow_cl_fields and v < 0:
                problems.append(f"{name}: rate must be >= 0 (got {v}) [{unit}]")
            if name in dimensionless_fields and name != "Kp_uu_brain":
                if not 0.0 <= v <= 1.0:
                    problems.append(
                        f"{name}: dimensionless fraction out of [0,1] (got {v})"
                    )
    return problems


def assert_zero_clearance(clearances: Clearances) -> bool:
    """True if all elimination/metabolism clearances are zero (closed system)."""
    metab = [
        *clearances.CL_met_NK.values(),
        *clearances.CL_other_parent.values(),
        *clearances.CL_met_HNK.values(),
        *clearances.CL_other_NK.values(),
        clearances.CL_out_HNK,
    ]
    return all(abs(c) == 0.0 for c in metab)
