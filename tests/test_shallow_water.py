"""Tests for the rotating shallow-water simulator."""

import numpy as np
import pytest

from geoflow.shallow_water import (RotatingShallowWater, initial_height,
                                   rossby_adjustment_check)

NX, NY, DX, DT = 160, 100, 4000.0, 150.0


def _mound_sim(**kwargs):
    sim = RotatingShallowWater(NX, NY, DX, DT, **kwargs)
    sim.set_initial_height(initial_height("mound", NX, NY, DX))
    return sim


def test_matches_textbook_rossby_adjustment():
    """Height profile and jet speed agree with Gill's analytic solution."""
    r = rossby_adjustment_check()
    assert r["eta_rel_error"] < 0.02                  # height profile within 2%
    assert abs(r["v_peak_error"]) < 0.08              # jet speed within 8%


def test_energy_is_conserved_without_damping():
    sim = _mound_sim()
    e0 = sim.energy()
    sim.run(int(1 * 86400 / DT))
    assert abs(sim.energy() / e0 - 1) < 1e-3


def test_flat_water_stays_flat():
    sim = RotatingShallowWater(NX, NY, DX, DT)
    sim.run(200)
    assert np.all(sim.eta == 0) and np.all(sim.u == 0) and np.all(sim.v == 0)


def test_unstable_time_step_is_refused():
    with pytest.raises(ValueError):
        RotatingShallowWater(NX, NY, DX, dt=2000.0)


def test_water_runs_downhill_first_then_turns_clockwise():
    """
    On the east side of a hill of water:
      - at first the flow is DOWNHILL (eastward, outwards)
      - later the flow is ALONG the contours: southward = clockwise around
        the hill in the Northern Hemisphere (high water on the right).
    """
    east = (NY // 2, NX // 2 + 17)                   # about one hill-radius east of centre
    sim = _mound_sim(sponge_cells=20)
    sim.run(int(3600 / DT))                          # one hour
    u, v = sim.velocity_at_centres()
    assert u[east] > 0 and abs(u[east]) > 3 * abs(v[east])      # downhill

    sim.run(int(2 * 86400 / DT))                     # two more days
    u, v = sim.velocity_at_centres()
    assert v[east] < 0 and abs(v[east]) > 3 * abs(u[east])      # along the contours


def test_settled_flow_is_in_geostrophic_balance_on_average():
    """
    After adjustment, f*v = g * d(eta)/dx (the formula from the notebook).

    The water keeps a small inertial oscillation (period 2*pi/f, about 17.5 h)
    that swings the instantaneous ratio between roughly 0.75 and 1.3, so the
    balance is checked on the average over one full oscillation.
    """
    sim = _mound_sim(sponge_cells=20)
    sim.run(int(3 * 86400 / DT))
    j = slice(NY // 2 - 8, NY // 2 + 9)
    i = slice(NX // 2 + 6, NX // 2 + 40)             # the east flank, where the slope is clear
    n = int(2 * np.pi / sim.f / DT)
    v_mean = np.zeros((NY, NX))
    gv_mean = np.zeros((NY, NX))
    for _ in range(n):
        sim.step()
        v_mean += sim.velocity_at_centres()[1] / n
        deta_dx = (np.roll(sim.eta, -1, axis=1) - np.roll(sim.eta, 1, axis=1)) / (2 * DX)
        gv_mean += (sim.g / sim.f) * deta_dx / n
    ratio = np.sum(v_mean[j, i] * gv_mean[j, i]) / np.sum(gv_mean[j, i] ** 2)
    assert 0.97 < ratio < 1.03
