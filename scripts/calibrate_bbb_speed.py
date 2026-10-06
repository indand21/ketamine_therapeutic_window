"""Calibrate bbb_speed_factor to the measured blood-effect-site equilibration.

The ratios of the four blood-brain transfer clearances are fixed by the unbound
partition coefficient Kp,uu = 0.6 (Moaddel 2023, paired cerebrospinal fluid).
Their magnitudes set how fast brain extracellular fluid tracks plasma, and were
previously unconstrained.

Olofsen et al. (Anesthesiology 2022;136:792-801, PMID 35188952) estimated a
blood-effect-site equilibration half-life of 8.3 min (95% CI 5.1 to 13.0) for
S-ketamine, for both the psychedelic (Bowdle external perception) and the
antinociceptive endpoint, in the same 17 volunteers whose arterial
concentrations calibrate layer 1 (Kamp 2020). t1/2ke0 is a lumped
plasma-to-effect delay, so assigning all of it to blood-brain transfer is an
upper bound on how slow that step can be; receptor binding in layer 2 adds a
further few tenths of a minute.

This script measures the model's plasma-to-brain-ECF equilibration half-time
under a constant plasma concentration and solves for the scale factor that
reproduces a target half-time.

Run with:  PYTHONPATH=. python scripts/calibrate_bbb_speed.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src.l1_pk.config import L1Parameters

TARGET_MIN = 8.3
TARGET_CI_MIN = (5.1, 13.0)
OUT = Path("results/bbb_speed_calibration.json")


def equilibration_half_time_min(params: L1Parameters, enantiomer: str = "S") -> float:
    """Half-time for brain ECF to reach steady state under constant plasma [min].

    Solves the two-compartment blood-brain chain (vascular, then extracellular)
    with the central concentration clamped at 1, exactly as layer 1 writes it.
    """
    cl = params.clearances
    cl_in = cl.transfer("CL_in")[enantiomer]
    cl_out = cl.transfer("CL_out")[enantiomer]
    cl_ei = cl.transfer("CL_ecf_in")[enantiomer]
    cl_eo = cl.transfer("CL_ecf_out")[enantiomer]
    v_vasc, v_ecf = params.volumes.V_vasc, params.volumes.V_ecf

    def rhs(_t, y):
        c_vasc, c_ecf = y
        return [
            (cl_in * 1.0 - cl_out * c_vasc - cl_ei * c_vasc + cl_eo * c_ecf) / v_vasc,
            (cl_ei * c_vasc - cl_eo * c_ecf) / v_ecf,
        ]

    t_end = max(2.0, 40.0 / cl.bbb_speed_factor / 60.0 * 6)
    sol = solve_ivp(rhs, [0.0, t_end], [0.0, 0.0], dense_output=True,
                    rtol=1e-11, atol=1e-13, method="LSODA")
    grid = np.linspace(0.0, t_end, 400001)
    c_ecf = sol.sol(grid)[1]
    css = c_ecf[-1]
    return float(grid[np.argmax(c_ecf >= 0.5 * css)] * 60.0)


def factor_for(target_min: float, base: L1Parameters) -> float:
    def err(f: float) -> float:
        p = dataclasses.replace(
            base, clearances=dataclasses.replace(base.clearances, bbb_speed_factor=f))
        return equilibration_half_time_min(p) - target_min

    return float(brentq(err, 1e-3, 100.0, xtol=1e-9))


def main() -> None:
    base = L1Parameters()
    t_default = equilibration_half_time_min(base)
    factor = factor_for(TARGET_MIN, base)
    lo = factor_for(TARGET_CI_MIN[1], base)   # slower target -> smaller factor
    hi = factor_for(TARGET_CI_MIN[0], base)

    checked = dataclasses.replace(
        base, clearances=dataclasses.replace(base.clearances, bbb_speed_factor=factor))
    cl = checked.clearances
    kp_uu = ((cl.transfer("CL_in")["S"] / cl.transfer("CL_out")["S"])
             * (cl.transfer("CL_ecf_in")["S"] / cl.transfer("CL_ecf_out")["S"]))

    out = {
        "target_t_half_ke0_min": TARGET_MIN,
        "target_95CI_min": list(TARGET_CI_MIN),
        "source": ("Olofsen E, Kamp J, Henthorn TK, van Velzen M, Niesters M, "
                   "Sarton E, Dahan A. Ketamine psychedelic and antinociceptive "
                   "effects are connected. Anesthesiology. 2022;136(5):792-801. "
                   "PMID 35188952; doi:10.1097/ALN.0000000000004176"),
        "model_t_half_default_min": round(t_default, 3),
        "bbb_speed_factor": round(factor, 4),
        "bbb_speed_factor_95CI": [round(lo, 4), round(hi, 4)],
        "fold_too_fast_before": round(TARGET_MIN / t_default, 2),
        "t_half_after_min": round(equilibration_half_time_min(checked), 3),
        "Kp_uu_after": round(kp_uu, 4),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        if k != "source":
            print(f"  {k}: {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
