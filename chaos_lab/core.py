"""Physics and numerical methods for the Nonlinear Dynamics & Chaos Lab.

The module keeps model equations independent from Streamlit so that every
experiment can be tested, reused, and exported without the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Callable

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class DrivenPendulumParams:
    gamma: float = 0.05
    omega0: float = 1.0
    drive_amplitude: float = 0.9
    drive_frequency: float = 0.65


@dataclass(frozen=True)
class KapitzaParams:
    gravity: float = 9.81
    length: float = 1.0
    pivot_amplitude: float = 0.12
    drive_frequency: float = 40.0
    damping: float = 0.08


@dataclass(frozen=True)
class DoublePendulumParams:
    mass1: float = 1.0
    mass2: float = 1.0
    length1: float = 1.0
    length2: float = 1.0
    gravity: float = 9.81
    damping: float = 0.0


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle to [-pi, pi)."""

    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _validate_double(params: DoublePendulumParams) -> None:
    for name, value in (
        ("mass1", params.mass1),
        ("mass2", params.mass2),
        ("length1", params.length1),
        ("length2", params.length2),
        ("gravity", params.gravity),
    ):
        _require_positive(name, value)
    if params.damping < 0 or not np.isfinite(params.damping):
        raise ValueError("damping must be non-negative and finite")


def driven_pendulum_ode(
    t: float,
    state: np.ndarray,
    params: DrivenPendulumParams,
) -> np.ndarray:
    theta, omega = state
    return np.asarray(
        [
            omega,
            -params.gamma * omega
            - params.omega0**2 * np.sin(theta)
            + params.drive_amplitude * np.cos(params.drive_frequency * t),
        ]
    )


def kapitza_pendulum_ode(
    t: float,
    state: np.ndarray,
    params: KapitzaParams,
) -> np.ndarray:
    theta, omega = state
    effective_acceleration = (
        params.gravity
        + params.pivot_amplitude * params.drive_frequency**2 * np.cos(params.drive_frequency * t)
    ) / params.length
    return np.asarray(
        [omega, -2.0 * params.damping * omega - effective_acceleration * np.sin(theta)]
    )


def double_pendulum_ode(
    _t: float,
    state: np.ndarray,
    params: DoublePendulumParams,
) -> np.ndarray:
    """Return [omega1, alpha1, omega2, alpha2]. Supports vectorized states."""

    _validate_double(params)
    theta1, omega1, theta2, omega2 = state
    m1, m2 = params.mass1, params.mass2
    l1, l2, g = params.length1, params.length2, params.gravity
    delta = theta1 - theta2
    common = 2.0 * m1 + m2 - m2 * np.cos(2.0 * delta)
    denominator1 = l1 * common
    denominator2 = l2 * common

    numerator1 = (
        -g * (2.0 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2.0 * theta2)
        - 2.0
        * np.sin(delta)
        * m2
        * (omega2**2 * l2 + omega1**2 * l1 * np.cos(delta))
    )
    numerator2 = 2.0 * np.sin(delta) * (
        omega1**2 * l1 * (m1 + m2)
        + g * (m1 + m2) * np.cos(theta1)
        + omega2**2 * l2 * m2 * np.cos(delta)
    )
    alpha1 = numerator1 / denominator1 - params.damping * omega1
    alpha2 = numerator2 / denominator2 - params.damping * omega2
    return np.asarray([omega1, alpha1, omega2, alpha2])


def rk4_step(
    function: Callable[..., np.ndarray],
    t: float,
    state: np.ndarray,
    dt: float,
    *args: object,
) -> np.ndarray:
    """Advance one classical fourth-order Runge-Kutta step."""

    _require_positive("dt", dt)
    k1 = function(t, state, *args)
    k2 = function(t + 0.5 * dt, state + 0.5 * dt * k1, *args)
    k3 = function(t + 0.5 * dt, state + 0.5 * dt * k2, *args)
    k4 = function(t + dt, state + dt * k3, *args)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_fixed_step(
    function: Callable[..., np.ndarray],
    initial_state: np.ndarray,
    *,
    duration: float,
    dt: float,
    args: tuple[object, ...] = (),
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a state with fixed-step RK4 and return every state."""

    _require_positive("duration", duration)
    _require_positive("dt", dt)
    steps = int(np.ceil(duration / dt))
    if steps > 2_000_000:
        raise ValueError("requested trajectory exceeds the 2,000,000-step safety limit")
    actual_dt = duration / steps
    state = np.asarray(initial_state, dtype=float).copy()
    times = np.linspace(0.0, duration, steps + 1)
    states = np.empty((steps + 1,) + state.shape, dtype=float)
    states[0] = state
    stride = max(1, steps // 100)
    for index in range(1, steps + 1):
        state = rk4_step(function, times[index - 1], state, actual_dt, *args)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"non-finite state produced at step {index}")
        states[index] = state
        if progress_callback and (index == steps or index % stride == 0):
            progress_callback(index, steps)
    return times, states


def driven_poincare_scan(
    amplitudes: np.ndarray,
    *,
    gamma: float = 0.05,
    omega0: float = 1.0,
    drive_frequency: float = 0.65,
    theta0: float = 0.1,
    omega_initial: float = 0.0,
    periods: int = 500,
    discard_periods: int = 200,
    steps_per_period: int = 120,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray]:
    """Stroboscopically sample driven pendula once per drive period."""

    values = np.asarray(amplitudes, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("amplitudes must be a finite, non-empty one-dimensional array")
    if np.any(values < 0):
        raise ValueError("drive amplitudes must be non-negative")
    _require_positive("drive_frequency", drive_frequency)
    _require_positive("omega0", omega0)
    if gamma < 0 or discard_periods < 0 or periods <= discard_periods:
        raise ValueError("invalid damping or transient-period settings")
    if steps_per_period < 20:
        raise ValueError("steps_per_period must be at least 20")

    period = 2.0 * np.pi / drive_frequency
    dt = period / steps_per_period
    state = np.asarray([np.full(values.size, theta0), np.full(values.size, omega_initial)])
    theta_samples: list[np.ndarray] = []
    omega_samples: list[np.ndarray] = []
    amplitude_samples: list[np.ndarray] = []
    time = 0.0
    total_steps = periods * steps_per_period
    stride = max(1, total_steps // 100)
    params = DrivenPendulumParams(gamma, omega0, 0.0, drive_frequency)

    def vector_ode(t: float, y: np.ndarray, base: DrivenPendulumParams) -> np.ndarray:
        theta, omega = y
        return np.asarray(
            [
                omega,
                -base.gamma * omega
                - base.omega0**2 * np.sin(theta)
                + values * np.cos(base.drive_frequency * t),
            ]
        )

    for step in range(1, total_steps + 1):
        state = rk4_step(vector_ode, time, state, dt, params)
        time += dt
        if step % steps_per_period == 0 and step // steps_per_period > discard_periods:
            amplitude_samples.append(values.copy())
            theta_samples.append(np.asarray(wrap_angle(state[0])))
            omega_samples.append(state[1].copy())
        if progress_callback and (step == total_steps or step % stride == 0):
            progress_callback(step, total_steps)

    return {
        "drive_amplitude": np.concatenate(amplitude_samples),
        "theta": np.concatenate(theta_samples),
        "omega": np.concatenate(omega_samples),
        "period": np.full(sum(item.size for item in amplitude_samples), period),
    }


def kapitza_stability_scan(
    amplitudes: np.ndarray,
    *,
    gravity: float = 9.81,
    length: float = 1.0,
    drive_frequency: float = 40.0,
    damping: float = 0.08,
    initial_offset: float = 0.05,
    periods: int = 240,
    discard_periods: int = 160,
    steps_per_period: int = 80,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray]:
    """Scan pivot amplitude and sample the damped response each drive period."""

    values = np.asarray(amplitudes, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError("amplitudes must be a non-negative one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("amplitudes must be finite")
    for name, value in (("gravity", gravity), ("length", length), ("drive_frequency", drive_frequency)):
        _require_positive(name, value)
    if damping < 0 or periods <= discard_periods or steps_per_period < 20:
        raise ValueError("invalid damping, period, or resolution settings")

    period = 2.0 * np.pi / drive_frequency
    dt = period / steps_per_period
    state = np.asarray(
        [np.full(values.size, np.pi - initial_offset), np.zeros(values.size)]
    )
    theta_samples: list[np.ndarray] = []
    amplitude_samples: list[np.ndarray] = []
    time = 0.0
    total_steps = periods * steps_per_period
    stride = max(1, total_steps // 100)

    def vector_ode(t: float, y: np.ndarray) -> np.ndarray:
        theta, omega = y
        acceleration = (
            gravity + values * drive_frequency**2 * np.cos(drive_frequency * t)
        ) / length
        return np.asarray([omega, -2.0 * damping * omega - acceleration * np.sin(theta)])

    for step in range(1, total_steps + 1):
        state = rk4_step(vector_ode, time, state, dt)
        time += dt
        if step % steps_per_period == 0 and step // steps_per_period > discard_periods:
            amplitude_samples.append(values.copy())
            theta_samples.append(np.asarray(wrap_angle(state[0])))
        if progress_callback and (step == total_steps or step % stride == 0):
            progress_callback(step, total_steps)

    threshold = kapitza_critical_amplitude(gravity, length, drive_frequency)
    sampled_amplitude = np.concatenate(amplitude_samples)
    sampled_theta = np.concatenate(theta_samples)
    upright_deviation = np.abs(np.asarray(wrap_angle(sampled_theta - np.pi)))
    return {
        "pivot_amplitude": sampled_amplitude,
        "theta": sampled_theta,
        "upright_deviation": upright_deviation,
        "critical_amplitude": np.full(sampled_amplitude.size, threshold),
    }


def kapitza_critical_amplitude(
    gravity: float,
    length: float,
    drive_frequency: float,
    *,
    precision_digits: int = 80,
) -> float:
    """Return the high-frequency approximate stabilization threshold."""

    for name, value in (("gravity", gravity), ("length", length), ("drive_frequency", drive_frequency)):
        _require_positive(name, value)
    if not isinstance(precision_digits, int) or not 20 <= precision_digits <= 500:
        raise ValueError("precision_digits must be an integer from 20 to 500")
    with mp.workdps(precision_digits):
        return float(mp.sqrt(2 * mp.mpf(gravity) * mp.mpf(length)) / mp.mpf(drive_frequency))


def simulate_double_pendulum(
    initial_state: np.ndarray,
    params: DoublePendulumParams,
    *,
    duration: float,
    dt: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_double(params)
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain four finite values")
    return simulate_fixed_step(
        double_pendulum_ode,
        state,
        duration=duration,
        dt=dt,
        args=(params,),
        progress_callback=progress_callback,
    )


def simulate_double_pendulum_adaptive(
    initial_state: np.ndarray,
    params: DoublePendulumParams,
    *,
    duration: float,
    samples: int = 1000,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """High-accuracy DOP853 trajectory used as an adaptive reference."""

    _validate_double(params)
    _require_positive("duration", duration)
    if samples < 2 or rtol <= 0 or atol <= 0:
        raise ValueError("invalid adaptive-solver settings")
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain four finite values")
    times = np.linspace(0.0, duration, samples)
    solution = solve_ivp(
        lambda t, y: double_pendulum_ode(t, y, params),
        (0.0, duration),
        initial,
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y.T


def double_pendulum_energy(
    states: np.ndarray,
    params: DoublePendulumParams,
) -> np.ndarray:
    """Mechanical energy of an undriven double pendulum."""

    _validate_double(params)
    values = np.asarray(states, dtype=float)
    theta1, omega1, theta2, omega2 = np.moveaxis(values, -1, 0)
    m1, m2 = params.mass1, params.mass2
    l1, l2, g = params.length1, params.length2, params.gravity
    kinetic = (
        0.5 * (m1 + m2) * l1**2 * omega1**2
        + 0.5 * m2 * l2**2 * omega2**2
        + m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )
    potential = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
    return kinetic + potential


def double_pendulum_cartesian(
    states: np.ndarray,
    params: DoublePendulumParams,
) -> dict[str, np.ndarray]:
    values = np.asarray(states, dtype=float)
    theta1, theta2 = values[..., 0], values[..., 2]
    x1 = params.length1 * np.sin(theta1)
    y1 = -params.length1 * np.cos(theta1)
    x2 = x1 + params.length2 * np.sin(theta2)
    y2 = y1 - params.length2 * np.cos(theta2)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def double_pendulum_mass_response(
    mass1_values: np.ndarray,
    base_params: DoublePendulumParams,
    *,
    initial_state: np.ndarray,
    duration: float = 20.0,
    dt: float = 0.01,
    transient_fraction: float = 0.4,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray]:
    """Scan upper mass and record post-transient local maxima of theta1.

    This is deliberately named a response spectrum rather than a bifurcation
    diagram: an undriven conservative double pendulum has no attracting steady
    state, so peak samples alone do not prove a dynamical bifurcation.
    """

    values = np.asarray(mass1_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ValueError("mass1_values must be a positive one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("mass1_values must be finite")
    if not 0 <= transient_fraction < 1:
        raise ValueError("transient_fraction must be in [0, 1)")
    _require_positive("duration", duration)
    _require_positive("dt", dt)
    estimated_steps = values.size * int(np.ceil(duration / dt))
    if estimated_steps > 5_000_000:
        raise ValueError("mass-response request exceeds the 5,000,000-step safety limit")
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain four finite values")
    mass_points: list[np.ndarray] = []
    peak_points: list[np.ndarray] = []
    peak_counts = np.zeros(values.size, dtype=int)
    for index, mass1 in enumerate(values):
        params = replace(base_params, mass1=float(mass1))
        _, states = simulate_double_pendulum(initial, params, duration=duration, dt=dt)
        start = int(states.shape[0] * transient_fraction)
        omega = states[:, 1]
        maxima = np.flatnonzero((omega[:-1] > 0) & (omega[1:] <= 0)) + 1
        maxima = maxima[maxima >= max(start, 1)]
        if maxima.size:
            previous = maxima - 1
            denominator = omega[previous] - omega[maxima]
            fraction = np.divide(
                omega[previous],
                denominator,
                out=np.zeros_like(denominator),
                where=np.abs(denominator) > np.finfo(float).tiny,
            )
            interpolated_theta = states[previous, 0] + fraction * (
                states[maxima, 0] - states[previous, 0]
            )
            peaks = np.asarray(wrap_angle(interpolated_theta))
        else:
            peaks = np.asarray([], dtype=float)
        peak_counts[index] = peaks.size
        if peaks.size:
            mass_points.append(np.full(peaks.size, mass1))
            peak_points.append(peaks)
        if progress_callback:
            progress_callback(index + 1, values.size)
    return {
        "mass1": np.concatenate(mass_points) if mass_points else np.asarray([]),
        "theta1_peak": np.concatenate(peak_points) if peak_points else np.asarray([]),
        "scan_mass1": values,
        "peak_count": peak_counts,
    }


def lyapunov_benettin(
    initial_state: np.ndarray,
    params: DoublePendulumParams,
    *,
    duration: float = 40.0,
    dt: float = 0.005,
    perturbation: float = 1e-8,
    renormalization_interval: float = 0.25,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray | float]:
    """Estimate the largest finite-time Lyapunov exponent with renormalization."""

    _validate_double(params)
    _require_positive("duration", duration)
    _require_positive("dt", dt)
    _require_positive("perturbation", perturbation)
    _require_positive("renormalization_interval", renormalization_interval)
    base = np.asarray(initial_state, dtype=float).copy()
    if base.shape != (4,) or not np.all(np.isfinite(base)):
        raise ValueError("initial_state must contain four finite values")
    omega_scale = np.sqrt(params.gravity / max(params.length1, params.length2))
    direction = np.asarray([1.0, 0.0, 0.0, 0.0])
    perturbed = base + direction * perturbation
    total_steps = int(np.ceil(duration / dt))
    if total_steps > 2_000_000:
        raise ValueError("Lyapunov calculation exceeds the step safety limit")
    actual_dt = duration / total_steps
    interval_steps = max(1, int(round(renormalization_interval / actual_dt)))
    times: list[float] = []
    local_values: list[float] = []
    cumulative_values: list[float] = []
    cumulative_log = 0.0
    time = 0.0
    stride = max(1, total_steps // 100)

    for step in range(1, total_steps + 1):
        base = rk4_step(double_pendulum_ode, time, base, actual_dt, params)
        perturbed = rk4_step(double_pendulum_ode, time, perturbed, actual_dt, params)
        time += actual_dt
        if step % interval_steps == 0:
            raw_difference = perturbed - base
            raw_difference[[0, 2]] = wrap_angle(raw_difference[[0, 2]])
            scaled = raw_difference.copy()
            scaled[[1, 3]] /= omega_scale
            separation = float(np.linalg.norm(scaled))
            if separation <= np.finfo(float).tiny or not np.isfinite(separation):
                raise FloatingPointError("trajectory separation became unusable")
            interval_time = interval_steps * actual_dt
            growth_log = np.log(separation / perturbation)
            cumulative_log += growth_log
            times.append(time)
            local_values.append(growth_log / interval_time)
            cumulative_values.append(cumulative_log / time)
            normalized = scaled * (perturbation / separation)
            normalized[[1, 3]] *= omega_scale
            perturbed = base + normalized
        if progress_callback and (step == total_steps or step % stride == 0):
            progress_callback(step, total_steps)

    estimate = float(cumulative_values[-1]) if cumulative_values else np.nan
    return {
        "time": np.asarray(times),
        "local_exponent": np.asarray(local_values),
        "cumulative_exponent": np.asarray(cumulative_values),
        "estimate": estimate,
        "requested_dt": float(dt),
        "actual_dt": float(actual_dt),
    }


def lyapunov_convergence_scan(
    dt_values: np.ndarray,
    initial_state: np.ndarray,
    params: DoublePendulumParams,
    *,
    duration: float = 30.0,
    perturbation: float = 1e-8,
    renormalization_interval: float = 0.25,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray | float]:
    """Repeat the Benettin estimate across timesteps to expose numerical sensitivity."""

    values = np.asarray(dt_values, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("dt_values must contain at least two positive finite values")
    estimates: list[float] = []
    actual_dts: list[float] = []
    for index, value in enumerate(values):
        result = lyapunov_benettin(
            initial_state,
            params,
            duration=duration,
            dt=float(value),
            perturbation=perturbation,
            renormalization_interval=renormalization_interval,
        )
        estimates.append(float(result["estimate"]))
        actual_dts.append(float(result["actual_dt"]))
        if progress_callback:
            progress_callback(index + 1, values.size)
    estimates_array = np.asarray(estimates)
    finest = estimates_array[np.argmin(values)]
    spread = float(np.max(estimates_array) - np.min(estimates_array))
    relative_spread = spread / max(abs(float(finest)), np.finfo(float).tiny)
    return {
        "dt": values,
        "requested_dt": values,
        "actual_dt": np.asarray(actual_dts),
        "estimate": estimates_array,
        "finest_dt_estimate": float(finest),
        "absolute_spread": spread,
        "relative_spread": float(relative_spread),
    }


def flip_time_map(
    theta1_values: np.ndarray,
    theta2_values: np.ndarray,
    params: DoublePendulumParams,
    *,
    t_max: float = 60.0,
    dt: float = 0.02,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray]:
    """Return the first time either arm rotates beyond its initial +/-pi branch."""

    _validate_double(params)
    if params.damping != 0:
        raise ValueError("flip maps currently require zero damping")
    values1 = np.asarray(theta1_values, dtype=float)
    values2 = np.asarray(theta2_values, dtype=float)
    if values1.ndim != 1 or values2.ndim != 1 or values1.size < 2 or values2.size < 2:
        raise ValueError("each angle axis must contain at least two points")
    if not np.all(np.isfinite(values1)) or not np.all(np.isfinite(values2)):
        raise ValueError("angle axes must be finite")
    _require_positive("t_max", t_max)
    _require_positive("dt", dt)
    steps = int(np.ceil(t_max / dt))
    actual_dt = t_max / steps
    system_steps = values1.size * values2.size * steps
    if system_steps > 75_000_000:
        raise ValueError("flip-map request exceeds the 75,000,000 system-step safety limit")

    grid1, grid2 = np.meshgrid(values1, values2, indexing="ij")
    count = grid1.size
    state = np.asarray(
        [grid1.ravel(), np.zeros(count), grid2.ravel(), np.zeros(count)]
    )
    flip_times = np.full(count, np.inf)
    initial_flipped = (np.abs(state[0]) >= np.pi) | (np.abs(state[2]) >= np.pi)
    flip_times[initial_flipped] = 0.0
    alive = np.arange(count)[~initial_flipped]
    state = state[:, ~initial_flipped]
    time = 0.0
    stride = max(1, steps // 100)
    for step in range(1, steps + 1):
        if alive.size == 0:
            if progress_callback:
                progress_callback(steps, steps)
            break
        state = rk4_step(double_pendulum_ode, time, state, actual_dt, params)
        time += actual_dt
        newly_flipped = (np.abs(state[0]) >= np.pi) | (np.abs(state[2]) >= np.pi)
        if np.any(newly_flipped):
            flip_times[alive[newly_flipped]] = time
            keep = ~newly_flipped
            state = state[:, keep]
            alive = alive[keep]
        if progress_callback and (step == steps or step % stride == 0):
            progress_callback(step, steps)
        if alive.size == 0:
            if progress_callback:
                progress_callback(steps, steps)
            break
    return {
        "theta1": values1,
        "theta2": values2,
        "flip_time": flip_times.reshape(values1.size, values2.size),
        "t_max": np.asarray(t_max),
        "requested_dt": np.asarray(dt),
        "actual_dt": np.asarray(actual_dt),
        "dt": np.asarray(actual_dt),
    }


def step_convergence_diagnostic(
    initial_state: np.ndarray,
    params: DoublePendulumParams,
    *,
    duration: float = 5.0,
    dt: float = 0.01,
) -> dict[str, float]:
    """Compare RK4 at dt and dt/2 and report conservative energy drift."""

    if params.damping != 0:
        raise ValueError("energy convergence diagnostic requires zero damping")
    _, coarse = simulate_double_pendulum(initial_state, params, duration=duration, dt=dt)
    _, fine = simulate_double_pendulum(initial_state, params, duration=duration, dt=dt / 2.0)
    difference = coarse[-1] - fine[-1]
    difference[[0, 2]] = wrap_angle(difference[[0, 2]])
    omega_scale = np.sqrt(params.gravity / max(params.length1, params.length2))
    scaled = difference.copy()
    scaled[[1, 3]] /= omega_scale
    coarse_energy = double_pendulum_energy(coarse, params)
    fine_energy = double_pendulum_energy(fine, params)
    coarse_drift = np.max(np.abs(coarse_energy - coarse_energy[0])) / max(
        abs(coarse_energy[0]), np.finfo(float).tiny
    )
    fine_drift = np.max(np.abs(fine_energy - fine_energy[0])) / max(
        abs(fine_energy[0]), np.finfo(float).tiny
    )
    return {
        "final_state_difference": float(np.linalg.norm(scaled)),
        "coarse_relative_energy_drift": float(coarse_drift),
        "fine_relative_energy_drift": float(fine_drift),
        "energy_drift_improved": bool(fine_drift <= coarse_drift),
    }
