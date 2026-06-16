"""Mass-conservation verification (Task 5.1, TechSpec §7, Protocol §10).

Closed-system test: with all metabolism/elimination clearances zeroed, the
total tracked drug mass must equal the cumulative dosed mass at every time
point, within solver tolerance.
"""

import numpy as np
import pytest

from src.l1_pk import L1Model, DosingRegimen, DoseEvent, InfusionSegment
from src.l1_pk.config import default_parameters
from src.verification import mass_conservation_residual
from src.verification.checks import assert_zero_clearance


@pytest.fixture
def closed_model():
    params = default_parameters().with_zero_clearance()
    assert assert_zero_clearance(params.clearances)
    return L1Model(params)


def test_bolus_only_mass_conserved(closed_model):
    regimen = DosingRegimen(
        boluses=[DoseEvent(time=0.0, amount=100.0)], s_fraction=0.5
    )
    t_end = 12.0
    t_eval = np.linspace(0.0, t_end, 400)
    result = closed_model.simulate(regimen, t_end, t_eval=t_eval)

    assert result.success, result.message
    dosed = regimen.total_dosed_mass(t_end)
    assert dosed == pytest.approx(100.0, rel=1e-9)
    residual = mass_conservation_residual(result, regimen)
    assert residual < 1e-4, f"mass drift {residual:.3e} mg exceeds tolerance"


def test_infusion_mass_conserved_after_stop(closed_model):
    # Infuse 10 mg/h for 4 h (= 40 mg total), then observe out to 12 h.
    regimen = DosingRegimen(
        infusions=[InfusionSegment(start=0.0, end=4.0, rate=10.0)],
        s_fraction=0.5,
    )
    t_end = 12.0
    t_eval = np.linspace(0.0, t_end, 500)
    result = closed_model.simulate(regimen, t_end, t_eval=t_eval)

    assert result.success, result.message
    dosed = regimen.total_dosed_mass(t_end)
    assert dosed == pytest.approx(40.0, rel=1e-9)
    # After infusion stops, total mass plateaus at the dosed amount.
    final_mass = result.total_mass()[-1]
    assert final_mass == pytest.approx(40.0, abs=1e-4)


def test_s_ketamine_regimen_no_r_mass(closed_model):
    # Pure S-ketamine: u_R = 0, so no R-species mass may appear.
    regimen = DosingRegimen(
        boluses=[DoseEvent(time=0.0, amount=50.0)], s_fraction=1.0
    )
    t_end = 8.0
    result = closed_model.simulate(regimen, t_end,
                                   t_eval=np.linspace(0, t_end, 200))
    assert result.success, result.message

    r_total = sum(
        result.amount(sp, cp).max()
        for sp in ("KET_R", "NK_R")
        for cp in ("cen", "per", "vasc", "ecf")
    )
    assert r_total == pytest.approx(0.0, abs=1e-9)
    # All dosed mass conserved within the S-side cells.
    assert result.total_mass()[-1] == pytest.approx(50.0, abs=1e-4)
