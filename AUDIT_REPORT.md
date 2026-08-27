# Audit report — Nonlinear Dynamics & Chaos Lab v1.1.0

## Scope

The v1.0.0 package was unpacked and its numerical core, Streamlit interface, launchers, tests, configuration files, and GitHub workflow were reviewed before modification.

## Important findings and changes

1. **Integration-window semantics** — v1.0.0 could overshoot requested end times in fixed-step trajectory, Lyapunov, and flip-map paths when the requested duration was not an integer multiple of `dt`. v1.1.0 uses a uniform `actual_dt <= requested_dt` so requested windows are not exceeded.
2. **Mass-response turning points** — maxima were assigned to the first stored sample after an angular-velocity sign change. v1.1.0 interpolates the zero crossing of `ω₁` and evaluates the peak angle between the two samples.
3. **Kapitza interpretation** — the analytical high-frequency threshold was already correct. v1.1.0 adds `upright_deviation = |wrap(theta - pi)|` so stability around the inverted state can be summarized directly.
4. **Lyapunov reliability** — the Benettin renormalized estimate was already substantially better than a one-shot separation fit. v1.1.0 adds a multi-`dt` sensitivity scan because one positive finite-time exponent can still be resolution-sensitive.
5. **Data presentation** — long results are now hidden by default. Plots come first, then a short key-results table; full DataFrames and CSV exports are created on demand.
6. **Flip-map boundary semantics** — initial conditions already on `±π` are now recorded with first-flip time `0` instead of waiting until the first integration step.
7. **Launcher robustness** — fixed port 8501 could fail when another Streamlit process was running. Both launchers now search ports 8501–8520.
8. **Configuration import** — imported JSON now receives stronger range and relationship checks before values enter session state.
9. **GitHub upload compatibility** — the active Actions workflow was moved to `docs/github-actions/tests.yml.example`. This avoids the GitHub PAT error that occurs when a token may push code but lacks the special workflow permission.

## Physics semantics retained

- The driven-pendulum scan remains stroboscopic and uses drive amplitude as its horizontal scan variable.
- The Kapitza threshold remains explicitly labeled as a high-frequency approximation.
- The conservative double-pendulum peak scan is called a response spectrum, not automatically a bifurcation diagram.
- The 3D visualization remains `(x₂, y₂, time)`, not a claim of out-of-plane motion.
- The first-flip map remains a finite-window diagnostic.

## Remaining limitations

- Classical RK4 is not symplectic; long conservative chaotic trajectories still require timestep-refinement checks.
- The phenomenological double-pendulum damping term is a model choice, not a detailed bearing/friction model.
- A finite-time Lyapunov estimate depends on trajectory, time window, state-space scaling, perturbation size, renormalization interval, and timestep.
- Fine flip-map boundaries should not be interpreted without resolution refinement.
- Streamlit AppTest was not executed in the isolated audit container because Streamlit was unavailable there; the app source compiled successfully and the shipped requirements install Streamlit in a normal environment.
