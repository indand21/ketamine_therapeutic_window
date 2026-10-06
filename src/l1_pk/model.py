"""L1 PBPK/PK state-equation engine (TechSpec §3).

Tracks four species (S-/R-ketamine, norketamine, (2R,6R)-HNK) across four
compartments (central, peripheral, brain-vascular, brain-ECF) with explicit
BBB transfer. The brain-ECF concentration is the PD driver exported to L2
(C_brain := C_ecf, TechSpec §3.3).

State is stored as amounts [mg] per (species, compartment) cell; concentrations
[mg/L] are amount / volume. Working in amounts makes the closed-system
mass-conservation check exact (TechSpec §7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from src.l1_pk.config import ENANTIOMERS, L1Parameters
from src.l1_pk.dosing import DosingRegimen

# Tracked species. Parent ketamine and NK are enantiomer-resolved; (2R,6R)-HNK
# is a single terminal species formed from R-NK (TechSpec §3.2).
SPECIES = ("KET_S", "KET_R", "NK_S", "NK_R", "HNK")
# Parent/NK live in cen+per+vasc+ecf; HNK is tracked in cen only (terminal).
COMPARTMENTS = ("cen", "per", "vasc", "ecf")

# (species, compartment) cells that carry state, in fixed order.
_CELLS: tuple[tuple[str, str], ...] = tuple(
    (sp, cp)
    for sp in SPECIES
    for cp in (COMPARTMENTS if sp != "HNK" else ("cen",))
)
_INDEX: dict[tuple[str, str], int] = {cell: i for i, cell in enumerate(_CELLS)}
N_STATES = len(_CELLS)


def _vol(params: L1Parameters, comp: str) -> float:
    v = params.volumes
    return {"cen": v.V_cen, "per": v.V_per, "vasc": v.V_vasc, "ecf": v.V_ecf}[comp]


@dataclass
class SimulationResult:
    """Container for a solved L1 trajectory."""

    t: np.ndarray                       # time grid [h]
    amounts: np.ndarray                 # (N_STATES, n_t) amounts [mg]
    params: L1Parameters
    success: bool
    message: str

    def amount(self, species: str, comp: str) -> np.ndarray:
        return self.amounts[_INDEX[(species, comp)], :]

    def concentration(self, species: str, comp: str) -> np.ndarray:
        return self.amount(species, comp) / _vol(self.params, comp)

    def brain_ecf(self, species: str) -> np.ndarray:
        """C_brain := C_ecf export to L2 (TechSpec §3.3)."""
        return self.concentration(species, "ecf")

    def total_mass(self) -> np.ndarray:
        """Total drug mass [mg] summed over all cells at each time point."""
        return self.amounts.sum(axis=0)


class L1Model:
    """Assemble and integrate the L1 ODE system."""

    def __init__(self, params: L1Parameters) -> None:
        self.params = params

    def _rhs(self, t: float, y: np.ndarray, regimen: DosingRegimen) -> np.ndarray:
        p = self.params
        cl = p.clearances
        dy = np.zeros_like(y)

        def C(sp: str, cp: str) -> float:
            return y[_INDEX[(sp, cp)]] / _vol(p, cp)

        def add(sp: str, cp: str, rate: float) -> None:
            dy[_INDEX[(sp, cp)]] += rate

        for e in ENANTIOMERS:
            ket, nk = f"KET_{e}", f"NK_{e}"
            # Concentrations needed for the flux terms.
            c_ket_cen, c_ket_per = C(ket, "cen"), C(ket, "per")
            c_ket_vasc, c_ket_ecf = C(ket, "vasc"), C(ket, "ecf")
            c_nk_cen, c_nk_per = C(nk, "cen"), C(nk, "per")
            c_nk_vasc, c_nk_ecf = C(nk, "vasc"), C(nk, "ecf")

            # --- Parent ketamine (TechSpec §3.2) ---
            dist = p.flows.Q_per * (c_ket_per - c_ket_cen)
            # Blood-brain transfer scaled by bbb_speed_factor (Kp,uu invariant).
            cl_in, cl_out = cl.transfer("CL_in")[e], cl.transfer("CL_out")[e]
            cl_ei, cl_eo = cl.transfer("CL_ecf_in")[e], cl.transfer("CL_ecf_out")[e]
            bbb = -cl_in * c_ket_cen + cl_out * c_ket_vasc
            met_nk = cl.CL_met_NK[e] * c_ket_cen
            elim_other_parent = cl.CL_other_parent[e] * c_ket_cen
            add(ket, "cen", regimen.input_rate(t, e) + dist + bbb - met_nk - elim_other_parent)
            add(ket, "per", -dist)
            # BBB chain cen -> vasc -> ecf (reversible; TechSpec §3.2-§3.3).
            v_in = cl_in * c_ket_cen - cl_out * c_ket_vasc
            ecf_flux = cl_ei * c_ket_vasc - cl_eo * c_ket_ecf
            add(ket, "vasc", v_in - ecf_flux)
            add(ket, "ecf", ecf_flux)

            # --- Norketamine (TechSpec §3.2) ---
            nk_dist = p.flows.Q_per * (c_nk_per - c_nk_cen)
            nk_bbb = -cl_in * c_nk_cen + cl_out * c_nk_vasc
            form_nk = p.fractions.f_m * met_nk
            elim_hnk = cl.CL_met_HNK[e] * c_nk_cen
            elim_other = cl.CL_other_NK[e] * c_nk_cen
            add(nk, "cen", form_nk + nk_dist + nk_bbb - elim_hnk - elim_other)
            add(nk, "per", -nk_dist)
            nk_v_in = cl_in * c_nk_cen - cl_out * c_nk_vasc
            nk_ecf = cl_ei * c_nk_vasc - cl_eo * c_nk_ecf
            add(nk, "vasc", nk_v_in - nk_ecf)
            add(nk, "ecf", nk_ecf)

            # --- (2R,6R)-HNK from R-NK only (TechSpec §3.2) ---
            if e == "R":
                add("HNK", "cen", p.fractions.f_m_HNK * elim_hnk
                    - cl.CL_out_HNK * C("HNK", "cen"))

        return dy

    def simulate(
        self,
        regimen: DosingRegimen,
        t_end: float,
        t_eval: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        method: str = "LSODA",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        max_step: float | None = None,
    ) -> SimulationResult:
        """Integrate the L1 system over [0, t_end] (TechSpec §3, Task 2.4).

        When the regimen contains boluses (narrow finite-rate pulses), the
        solver step is capped so the adaptive integrator cannot step over the
        pulse and miss the mass input. Callers may override via ``max_step``.
        """
        if y0 is None:
            y0 = np.zeros(N_STATES)
        if max_step is None:
            max_step = (
                regimen.bolus_width / 4.0 if regimen.has_boluses() else np.inf
            )
        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, regimen),
            t_span=(0.0, t_end),
            y0=y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=False,
        )
        return SimulationResult(
            t=sol.t, amounts=sol.y, params=self.params,
            success=sol.success, message=sol.message,
        )
