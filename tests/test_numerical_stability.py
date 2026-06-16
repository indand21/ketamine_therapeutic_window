"""Numerical-stability & unit-consistency verification (Task 5.2).

Covers dimensional consistency of the configuration, ODE solver convergence
under tolerance refinement, RHS shape/finiteness, and seeded reproducibility
(TechSpec §7, Protocol §10-§11).
"""

import numpy as np
import pytest

from src.l1_pk import L1Model, DosingRegimen, DoseEvent
from src.l1_pk.config import default_parameters, ENANTIOMERS
from src.l1_pk.model import N_STATES
from src.verification import check_unit_consistency


@pytest.fixture
def open_model():
    return L1Model(default_parameters())


@pytest.fixture
def regimen():
    return DosingRegimen(boluses=[DoseEvent(time=0.0, amount=100.0)],
                         s_fraction=0.5)


def test_unit_consistency_default_params():
    problems = check_unit_consistency(default_parameters())
    assert problems == [], "dimensional problems: " + "; ".join(problems)


def test_unit_consistency_flags_bad_volume():
    from src.l1_pk.config import CompartmentVolumes, L1Parameters
    bad = L1Parameters(volumes=CompartmentVolumes(V_ecf=-1.0))
    problems = check_unit_consistency(bad)
    assert any("V_ecf" in p for p in problems)


def test_rhs_shape_and_finiteness(open_model, regimen):
    y = np.ones(N_STATES)
    dy = open_model._rhs(1.0, y, regimen)
    assert dy.shape == (N_STATES,)
    assert np.all(np.isfinite(dy))


def test_solver_convergence_under_tolerance_refinement(open_model, regimen):
    t_end = 12.0
    t_eval = np.linspace(0.0, t_end, 200)
    coarse = open_model.simulate(regimen, t_end, t_eval=t_eval,
                                 rtol=1e-5, atol=1e-7)
    fine = open_model.simulate(regimen, t_end, t_eval=t_eval,
                               rtol=1e-9, atol=1e-11)
    assert coarse.success and fine.success

    # Solutions must converge: refining tolerance changes the trajectory
    # by a small, bounded amount on the PD-driver channel.
    c_coarse = coarse.brain_ecf("KET_S")
    c_fine = fine.brain_ecf("KET_S")
    denom = np.max(np.abs(c_fine)) + 1e-12
    rel_diff = np.max(np.abs(c_coarse - c_fine)) / denom
    assert rel_diff < 1e-3, f"solver not converged: rel diff {rel_diff:.3e}"


def test_nonnegativity_of_concentrations(open_model, regimen):
    result = open_model.simulate(regimen, 24.0,
                                 t_eval=np.linspace(0, 24, 400))
    assert result.success
    for sp in ("KET_S", "KET_R", "NK_S", "NK_R"):
        for cp in ("cen", "per", "vasc", "ecf"):
            c = result.concentration(sp, cp)
            assert c.min() > -1e-6, f"{sp}/{cp} went negative"


def test_seeded_reproducibility(open_model, regimen):
    t_eval = np.linspace(0.0, 12.0, 200)
    r1 = open_model.simulate(regimen, 12.0, t_eval=t_eval)
    r2 = open_model.simulate(regimen, 12.0, t_eval=t_eval)
    assert np.allclose(r1.amounts, r2.amounts, rtol=0, atol=0)


def test_hnk_forms_only_from_r_pathway(open_model):
    # Pure S-ketamine -> no R-NK -> no (2R,6R)-HNK formation.
    s_only = DosingRegimen(boluses=[DoseEvent(0.0, 100.0)], s_fraction=1.0)
    result = open_model.simulate(s_only, 24.0,
                                 t_eval=np.linspace(0, 24, 200))
    assert result.success
    assert result.amount("HNK", "cen").max() == pytest.approx(0.0, abs=1e-9)


def test_enantioselective_clearance_effect(open_model):
    # R has lower CL_met_NK than S by default -> R parent persists longer.
    racemic = DosingRegimen(boluses=[DoseEvent(0.0, 100.0)], s_fraction=0.5)
    result = open_model.simulate(racemic, 24.0,
                                 t_eval=np.linspace(0, 24, 400))
    assert result.success
    auc_s = np.trapz(result.concentration("KET_S", "cen"), result.t)
    auc_r = np.trapz(result.concentration("KET_R", "cen"), result.t)
    assert auc_r > auc_s
