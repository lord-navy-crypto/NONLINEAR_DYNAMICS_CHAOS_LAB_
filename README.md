# Nonlinear Dynamics & Chaos Lab

**Version 1.1.0 — GitHub-ready computational-physics laboratory**

An interactive localhost platform reconstructed from `script33.ipynb` and expanded into a reusable nonlinear-dynamics workbench. It studies driven pendula, Kapitza stabilization, double-pendulum motion, mass-response structure, finite-time Lyapunov divergence, first-flip maps, and numerical reliability.

The project separates the physics/numerical core from the Streamlit interface. The current release also adopts a **trend-first** presentation rule: plots and physical trends appear first, a short key-results table follows, and complete numerical tables are generated only when the user requests them.

## What changed in v1.1.0

- Fixed-step double-pendulum integration now ends at the **exact requested final time** even when `duration / dt` is not an integer.
- Double-pendulum peak locations in the upper-mass scan use interpolated angular-velocity zero crossings instead of the nearest stored timestep.
- Kapitza scans now expose an explicit **upright-deviation** diagnostic in addition to the sampled angle.
- Added a **Lyapunov timestep-sensitivity scan** so a single positive finite-time estimate is not presented without a numerical-resolution check.
- Long outputs use **plots → key results → on-demand complete data**.
- macOS and Windows launchers automatically choose the first free localhost port from **8501–8520** instead of failing when 8501 is occupied.
- GitHub issue templates, contribution guidance, release checklist, security/scope notes, editor settings, and an optional CI workflow template are included.
- The GitHub Actions workflow is intentionally shipped as an example rather than an active workflow so a normal HTTPS push does not require the special PAT `workflow` permission.

## Physics and experiments

### Driven damped pendulum

The implemented model is

\[
\ddot\theta + \gamma\dot\theta + \omega_0^2\sin\theta
= A\cos(\Omega t).
\]

The amplitude scan is sampled **stroboscopically**, once per drive period after a configurable transient interval. Drive amplitude is the horizontal scan axis.

### Kapitza inverted pendulum

The vertically driven pivot model uses

\[
\ddot\theta + 2\beta\dot\theta +
\frac{g+a\Omega^2\cos(\Omega t)}{L}\sin\theta = 0.
\]

The high-frequency approximate inverted-state stabilization threshold is

\[
a_{\mathrm{crit}} \approx \frac{\sqrt{2gL}}{\Omega}.
\]

The threshold is an asymptotic guide, not an exact boundary for arbitrary frequency, damping, transient length, or initial condition.

### Double pendulum

The platform integrates the full nonlinear planar double-pendulum equations with configurable masses, lengths, gravity, and phenomenological angular damping. It reports angles, bob coordinates, mechanical energy, physical lower-bob paths, and a clearly labeled space-time curve `(x₂, y₂, t)`.

### Upper-mass response scan

Post-transient local maxima of `θ₁` are plotted against the scanned upper mass. The platform deliberately calls this a **mass-response spectrum**, not automatically a bifurcation diagram: the conservative double pendulum does not have the attracting steady state assumed in the usual dissipative bifurcation construction.

### Largest finite-time Lyapunov exponent

The Benettin procedure repeatedly renormalizes trajectory separation. Version 1.1.0 can repeat the estimate across multiple timesteps and report the estimate spread.

### First-flip map

The map records the first time either arm leaves its initial `±π` angular branch. Cells that do not flip inside the selected observation window are distinguished from actual measured flip times. Requests above the configured system-step safety limit are blocked.

## Numerical reliability

The core includes:

- classical RK4 integration;
- DOP853 adaptive-solver comparison;
- conservative mechanical-energy drift;
- `dt` versus `dt/2` refinement;
- exact endpoint handling;
- workload limits for long trajectories and fractal grids;
- finite-value and physical-parameter validation;
- Lyapunov timestep sensitivity.

A visually complicated trajectory is not automatically evidence of physical chaos. Important conclusions should survive timestep refinement, longer observation windows, and changes in transient removal.

## Quick start

### macOS

1. Extract the complete project folder.
2. Double-click `RUN_CHAOS_LAB.command`.
3. The launcher creates `.venv`, installs dependencies, finds a free port in `8501–8520`, and starts Streamlit.

If Finder blocks the launcher once:

```bash
chmod +x RUN_CHAOS_LAB.command
./RUN_CHAOS_LAB.command
```

### Windows

Double-click `RUN_CHAOS_LAB.bat`.

### Manual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Core and repository tests do not require the UI to be running. `tests/test_app.py` uses Streamlit's AppTest when Streamlit is installed.

## Data presentation

Large numerical results are intentionally not dumped onto the page by default. Each experiment follows:

1. trend / figure;
2. key physical or numerical quantities;
3. **More data / complete data** button;
4. full table and CSV download only after the button is enabled.

## Repository layout

```text
.
├── app.py
├── chaos_lab/
│   ├── __init__.py
│   └── core.py
├── presets/
├── tests/
├── tools/find_free_port.py
├── docs/
├── .github/ISSUE_TEMPLATE/
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── RELEASE_CHECKLIST.md
├── RUN_CHAOS_LAB.command
├── RUN_CHAOS_LAB.bat
├── requirements.txt
└── pyproject.toml
```

## Optional GitHub Actions

`docs/github-actions/tests.yml.example` contains the CI workflow. See `docs/GITHUB_ACTIONS_OPTIONAL.md` for activation instructions. It is not active by default so users pushing with an HTTPS PAT that lacks workflow permission are not blocked.

## Scope

This is research/education software, not safety-certified engineering software. A passing built-in validation suite establishes only the checks that were actually run; it does not prove every long-time chaotic boundary or every parameter choice is numerically converged.

No software license is added automatically in this package. Repository owners should choose and add a license deliberately before redistribution if desired.
