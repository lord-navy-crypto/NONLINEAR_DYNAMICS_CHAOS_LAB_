"""Reusable numerical core for the Nonlinear Dynamics & Chaos Lab."""

from .core import (
    DoublePendulumParams,
    DrivenPendulumParams,
    KapitzaParams,
    double_pendulum_cartesian,
    double_pendulum_energy,
    double_pendulum_mass_response,
    driven_poincare_scan,
    flip_time_map,
    kapitza_critical_amplitude,
    kapitza_stability_scan,
    lyapunov_benettin,
    lyapunov_convergence_scan,
    simulate_double_pendulum,
    simulate_double_pendulum_adaptive,
    step_convergence_diagnostic,
    wrap_angle,
)

__all__ = [
    "DoublePendulumParams",
    "DrivenPendulumParams",
    "KapitzaParams",
    "double_pendulum_cartesian",
    "double_pendulum_energy",
    "double_pendulum_mass_response",
    "driven_poincare_scan",
    "flip_time_map",
    "kapitza_critical_amplitude",
    "kapitza_stability_scan",
    "lyapunov_benettin",
    "lyapunov_convergence_scan",
    "simulate_double_pendulum",
    "simulate_double_pendulum_adaptive",
    "step_convergence_diagnostic",
    "wrap_angle",
]
