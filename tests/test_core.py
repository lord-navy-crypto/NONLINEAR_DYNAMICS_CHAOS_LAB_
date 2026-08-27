import numpy as np
import pytest

from chaos_lab import (
    DoublePendulumParams,
    double_pendulum_energy,
    double_pendulum_mass_response,
    driven_poincare_scan,
    flip_time_map,
    kapitza_critical_amplitude,
    kapitza_stability_scan,
    lyapunov_benettin,
    simulate_double_pendulum,
    simulate_double_pendulum_adaptive,
    step_convergence_diagnostic,
    wrap_angle,
)


def test_wrap_angle_range() -> None:
    wrapped = wrap_angle(np.asarray([-4 * np.pi, -np.pi, 0.0, np.pi, 4 * np.pi]))
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)


def test_kapitza_threshold_matches_formula() -> None:
    threshold = kapitza_critical_amplitude(9.81, 1.0, 40.0)
    assert threshold == pytest.approx(np.sqrt(2 * 9.81) / 40.0, rel=1e-14)


def test_driven_scan_uses_amplitude_axis_and_stroboscopic_rows() -> None:
    amplitudes = np.asarray([0.3, 0.9])
    result = driven_poincare_scan(
        amplitudes, periods=8, discard_periods=5, steps_per_period=40
    )
    assert result["drive_amplitude"].shape == (6,)
    assert set(result["drive_amplitude"]) == set(amplitudes)


def test_kapitza_scan_uses_pivot_amplitude_axis() -> None:
    amplitudes = np.linspace(0.0, 0.2, 4)
    result = kapitza_stability_scan(
        amplitudes, periods=8, discard_periods=5, steps_per_period=40
    )
    assert result["pivot_amplitude"].shape == (12,)
    assert np.all(result["critical_amplitude"] > 0)


def test_double_pendulum_energy_is_conserved_at_small_dt() -> None:
    params = DoublePendulumParams()
    state = np.asarray([1.2, 0.0, 0.7, 0.0])
    _, states = simulate_double_pendulum(state, params, duration=2.0, dt=0.002)
    energy = double_pendulum_energy(states, params)
    drift = np.max(np.abs(energy - energy[0])) / abs(energy[0])
    assert drift < 1e-7


def test_mass_response_uses_non_degenerate_initial_state() -> None:
    result = double_pendulum_mass_response(
        np.asarray([0.8, 1.2]),
        DoublePendulumParams(),
        initial_state=np.asarray([1.2, 0.0, 0.6, 0.0]),
        duration=3.0,
        dt=0.01,
        transient_fraction=0.2,
    )
    np.testing.assert_array_equal(result["scan_mass1"], np.asarray([0.8, 1.2]))
    assert result["peak_count"].shape == (2,)


def test_lyapunov_benettin_returns_finite_progression() -> None:
    result = lyapunov_benettin(
        np.asarray([1.2, 0.0, 0.7, 0.0]),
        DoublePendulumParams(),
        duration=2.0,
        dt=0.01,
        renormalization_interval=0.2,
    )
    assert len(result["time"]) == 10
    assert np.isfinite(result["estimate"])


def test_flip_map_shape_and_quiet_center() -> None:
    axis = np.linspace(-0.2, 0.2, 5)
    result = flip_time_map(axis, axis, DoublePendulumParams(), t_max=1.0, dt=0.02)
    assert result["flip_time"].shape == (5, 5)
    assert np.isinf(result["flip_time"][2, 2])
    assert float(result["t_max"]) == 1.0


def test_step_refinement_reduces_energy_drift() -> None:
    diagnostic = step_convergence_diagnostic(
        np.asarray([1.2, 0.0, 0.7, 0.0]),
        DoublePendulumParams(),
        duration=2.0,
        dt=0.02,
    )
    assert diagnostic["energy_drift_improved"]
    assert diagnostic["fine_relative_energy_drift"] < diagnostic["coarse_relative_energy_drift"]


def test_adaptive_reference_returns_requested_samples() -> None:
    times, states = simulate_double_pendulum_adaptive(
        np.asarray([1.0, 0.0, 0.5, 0.0]),
        DoublePendulumParams(),
        duration=1.0,
        samples=41,
    )
    assert times.shape == (41,)
    assert states.shape == (41, 4)


def test_invalid_physical_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_double_pendulum(
            np.zeros(4), DoublePendulumParams(mass1=-1), duration=1.0, dt=0.01
        )
    with pytest.raises(ValueError):
        kapitza_critical_amplitude(9.81, 1.0, 0.0)


def test_fixed_step_trajectory_ends_at_requested_duration() -> None:
    times, states = simulate_double_pendulum(
        np.asarray([1.0, 0.0, 0.5, 0.0]),
        DoublePendulumParams(),
        duration=1.003,
        dt=0.01,
    )
    assert times[-1] == pytest.approx(1.003, abs=1e-15)
    assert states.shape[0] == times.size


def test_kapitza_scan_reports_upright_deviation() -> None:
    result = kapitza_stability_scan(
        np.asarray([0.0, 0.15]), periods=8, discard_periods=5, steps_per_period=40
    )
    assert result["upright_deviation"].shape == result["theta"].shape
    assert np.all(result["upright_deviation"] >= 0)
    assert np.all(result["upright_deviation"] <= np.pi)


def test_lyapunov_timestep_sensitivity_is_finite() -> None:
    from chaos_lab import lyapunov_convergence_scan

    result = lyapunov_convergence_scan(
        np.asarray([0.02, 0.01]),
        np.asarray([1.2, 0.0, 0.7, 0.0]),
        DoublePendulumParams(),
        duration=1.0,
    )
    assert result["estimate"].shape == (2,)
    assert np.all(np.isfinite(result["estimate"]))
    assert np.isfinite(result["absolute_spread"])


def test_lyapunov_and_flip_integrators_do_not_overshoot_requested_window() -> None:
    lyap = lyapunov_benettin(
        np.asarray([1.0, 0.0, 0.5, 0.0]),
        DoublePendulumParams(),
        duration=1.003,
        dt=0.01,
        renormalization_interval=0.1,
    )
    assert lyap["actual_dt"] <= lyap["requested_dt"]
    assert np.all(lyap["time"] <= 1.003 + 1e-12)

    axis = np.linspace(-0.2, 0.2, 3)
    flip = flip_time_map(axis, axis, DoublePendulumParams(), t_max=1.003, dt=0.02)
    assert float(flip["actual_dt"]) <= 0.02
    finite = flip["flip_time"][np.isfinite(flip["flip_time"])]
    assert np.all(finite <= 1.003 + 1e-12)


def test_flip_map_marks_boundary_initial_conditions_at_zero_time() -> None:
    axis = np.asarray([-np.pi, 0.0, np.pi])
    result = flip_time_map(axis, axis, DoublePendulumParams(), t_max=0.5, dt=0.02)
    assert np.all(result["flip_time"][[0, 2], :] == 0.0)
    assert np.all(result["flip_time"][:, [0, 2]] == 0.0)
