# Validation results — Nonlinear Dynamics & Chaos Lab v1.1.0

Audit environment: Python 3.13.5.

## Automated checks

- Core + repository tests: **20 passed**
- Python compilation: **PASS**
- `pyproject.toml` parsing: **PASS**
- preset JSON parsing: **PASS**
- macOS launcher shell syntax: **PASS**
- Python wheel build from `pyproject.toml`: **PASS**
- dynamic port helper test: **PASS**

`tests/test_app.py` is retained for CI / local environments with Streamlit installed, but Streamlit was not available in the isolated audit container.

## Numerical spot checks

For an undamped double pendulum with representative state `[1.2, 0.3, 0.7, -0.2]`, a centered directional derivative of mechanical energy along the implemented ODE gave:

- `dE/dt ≈ 0.000000000000e+00`

For a trajectory requested to end at `2.003 s` with requested `dt=0.01 s`:

- final stored time: `2.003000000000 s`
- relative RK4 energy drift: `4.332998682807e-08`

Timestep refinement (`duration=4 s`, coarse `dt=0.02 s`):

- coarse relative energy drift: `1.982377646342e-06`
- fine relative energy drift: `8.025682350571e-08`
- refinement improved energy drift: **True**
- scaled coarse/fine final-state difference: `2.610976117827e-05`

Fixed RK4 (`dt≈0.005`) versus DOP853 at `4 s`:

- final-state norm difference: `4.090607542401e-07`

Kapitza threshold check for `g=9.81 m/s²`, `L=1 m`, `Ω=40 rad/s`:

- `a_crit = 1.107361729518e-01 m`

Stroboscopic shape check with 2 amplitudes, 8 periods, 5 discarded periods:

- returned rows: `6` = `2 × (8-5)`

Lyapunov timestep sensitivity for one representative trajectory (`duration=2 s`):

- `dt = [0.02, 0.01, 0.005] s`
- estimates ≈ `[0.50016959, 0.32801640, 0.32801536] s⁻¹`
- relative spread ≈ `0.52484`

The large spread in this short-window example is intentionally informative: it demonstrates why the new timestep-sensitivity view is necessary before interpreting a single finite-time exponent.
