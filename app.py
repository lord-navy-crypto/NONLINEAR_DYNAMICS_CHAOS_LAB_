from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chaos_lab import (
    DoublePendulumParams,
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
)


APP_VERSION = "1.1.0"


st.set_page_config(
    page_title="Nonlinear Dynamics & Chaos Lab",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(145deg,#f6f8fb 0%,#ffffff 45%,#f1f7f7 100%)}
    .hero {padding:1.45rem 1.7rem;border-radius:20px;color:white;
      background:linear-gradient(120deg,#151f30 0%,#4b2e83 55%,#007c83 100%);
      box-shadow:0 14px 34px rgba(21,31,48,.2);margin-bottom:1rem}
    .hero h1 {margin:0 0 .35rem;font-size:2.15rem}.hero p{margin:0;opacity:.93}
    .note {padding:.8rem 1rem;border-left:4px solid #007c83;border-radius:8px;
      background:#ecf8f7;margin:.5rem 0 1rem}
    div[data-testid="stMetric"] {background:white;border:1px solid #dce5eb;
      padding:.7rem;border-radius:14px;box-shadow:0 4px 14px rgba(21,31,48,.05)}
    </style>
    """,
    unsafe_allow_html=True,
)


def defaults() -> dict[str, object]:
    return {
        "m1": 1.0,
        "m2": 1.0,
        "l1": 1.0,
        "l2": 1.0,
        "gravity": 9.81,
        "theta1_0": 1.57079632679,
        "omega1_0": 0.0,
        "theta2_0": 1.27079632679,
        "omega2_0": 0.0,
        "trajectory_duration": 20.0,
        "trajectory_dt": 0.01,
        "drive_a_min": 0.2,
        "drive_a_max": 1.4,
        "drive_a_points": 80,
        "drive_periods": 360,
        "drive_discard": 220,
        "drive_steps_per_period": 100,
        "kapitza_a_max": 0.25,
        "kapitza_points": 100,
        "kapitza_frequency": 40.0,
        "kapitza_damping": 0.08,
        "kapitza_periods": 220,
        "kapitza_discard": 150,
        "mass_min": 0.2,
        "mass_max": 4.0,
        "mass_points": 45,
        "mass_duration": 15.0,
        "mass_dt": 0.015,
        "lyap_duration": 30.0,
        "lyap_dt": 0.005,
        "flip_resolution": 60,
        "flip_tmax": 50.0,
        "flip_dt": 0.02,
    }


def initialize() -> None:
    for key, value in defaults().items():
        st.session_state.setdefault(key, value)
    for key in ("drive_result", "kapitza_result", "trajectory_result", "mass_result", "lyap_result", "lyap_convergence_result", "flip_result", "validation_result"):
        st.session_state.setdefault(key, None)


def current_config() -> dict[str, object]:
    return {
        "schema": "nonlinear-dynamics-chaos-lab-config-v1",
        "app_version": APP_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **{key: st.session_state[key] for key in defaults()},
    }


def load_config(uploaded) -> None:
    if uploaded is None:
        return
    signature = (uploaded.name, uploaded.size)
    if st.session_state.get("config_signature") == signature:
        return
    try:
        raw = json.loads(uploaded.getvalue().decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a JSON object")
        candidate = defaults()
        for key, fallback in candidate.items():
            if key in raw:
                candidate[key] = type(fallback)(raw[key])
        positive_keys = (
            "m1", "m2", "l1", "l2", "gravity", "trajectory_duration", "trajectory_dt",
            "drive_a_max", "kapitza_a_max", "kapitza_frequency", "mass_min", "mass_max",
            "mass_duration", "mass_dt", "lyap_duration", "lyap_dt", "flip_tmax", "flip_dt",
        )
        if any(float(candidate[key]) <= 0 for key in positive_keys):
            raise ValueError("physical scales, durations, and timesteps must be positive")
        if float(candidate["drive_a_min"]) < 0 or float(candidate["kapitza_damping"]) < 0:
            raise ValueError("drive amplitude and damping must be non-negative")
        if int(candidate["drive_a_points"]) < 2 or int(candidate["kapitza_points"]) < 2 or int(candidate["mass_points"]) < 2 or int(candidate["flip_resolution"]) < 2:
            raise ValueError("scan resolutions must contain at least two points")
        if candidate["drive_a_min"] >= candidate["drive_a_max"] or candidate["mass_min"] >= candidate["mass_max"]:
            raise ValueError("scan minima must be smaller than scan maxima")
        if candidate["drive_discard"] >= candidate["drive_periods"] or candidate["kapitza_discard"] >= candidate["kapitza_periods"]:
            raise ValueError("discarded periods must be fewer than total periods")
        for key, value in candidate.items():
            st.session_state[key] = value
        for result_key in ("drive_result", "kapitza_result", "trajectory_result", "mass_result", "lyap_result", "lyap_convergence_result", "flip_result", "validation_result"):
            st.session_state[result_key] = None
        st.session_state.config_signature = signature
        st.success("Configuration loaded. Run an experiment to generate new results.")
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        st.error(f"Could not load configuration: {exc}")


def double_params(*, damping: float = 0.0) -> DoublePendulumParams:
    return DoublePendulumParams(
        mass1=float(st.session_state.m1),
        mass2=float(st.session_state.m2),
        length1=float(st.session_state.l1),
        length2=float(st.session_state.l2),
        gravity=float(st.session_state.gravity),
        damping=damping,
    )


def initial_state() -> np.ndarray:
    return np.asarray(
        [
            st.session_state.theta1_0,
            st.session_state.omega1_0,
            st.session_state.theta2_0,
            st.session_state.omega2_0,
        ],
        dtype=float,
    )


def progress_reporter(label: str):
    bar = st.progress(0, text=label)

    def update(done: int, total: int) -> None:
        bar.progress(done / total, text=f"{label}: {done}/{total}")

    return bar, update


def download_frame(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def key_results(rows: list[tuple[str, object, str]]) -> None:
    frame = pd.DataFrame(rows, columns=["Key quantity", "Value", "Unit / meaning"])
    st.dataframe(frame, width="stretch", hide_index=True, height=min(360, 38 * (len(frame) + 1)))


def complete_data_panel(
    key: str,
    frame_factory,
    filename: str,
    *,
    label: str = "More data / complete data",
) -> None:
    state_key = f"show_complete_{key}"
    st.session_state.setdefault(state_key, False)
    button_label = "Hide complete data" if st.session_state[state_key] else label
    if st.button(button_label, key=f"toggle_{key}"):
        st.session_state[state_key] = not st.session_state[state_key]
    if st.session_state[state_key]:
        frame = frame_factory()
        st.dataframe(frame, width="stretch", hide_index=True, height=min(900, 38 * (len(frame) + 1)))
        download_frame("Download complete CSV", frame, filename)


initialize()

st.markdown(
    """
    <section class="hero">
      <h1>Nonlinear Dynamics & Chaos Lab</h1>
      <p>Driven pendula, Kapitza stabilization, double-pendulum trajectories,
      Lyapunov divergence, fractal flip maps, and numerical compliance checks.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1 · Setup — Shared physical model")
    uploaded = st.file_uploader("Import configuration", type=["json"])
    load_config(uploaded)
    c1, c2 = st.columns(2)
    c1.number_input("m₁ (kg)", min_value=0.05, max_value=20.0, key="m1")
    c2.number_input("m₂ (kg)", min_value=0.05, max_value=20.0, key="m2")
    c1.number_input("L₁ (m)", min_value=0.05, max_value=20.0, key="l1")
    c2.number_input("L₂ (m)", min_value=0.05, max_value=20.0, key="l2")
    st.number_input("Gravity g (m/s²)", min_value=0.01, max_value=100.0, key="gravity")
    st.subheader("Initial state")
    c1, c2 = st.columns(2)
    c1.number_input("θ₁(0) (rad)", key="theta1_0", format="%.6f")
    c2.number_input("ω₁(0) (rad/s)", key="omega1_0", format="%.6f")
    c1.number_input("θ₂(0) (rad)", key="theta2_0", format="%.6f")
    c2.number_input("ω₂(0) (rad/s)", key="omega2_0", format="%.6f")
    st.caption("The default angles are intentionally asymmetric, avoiding the degenerate θ₁=θ₂ motion in the original notebook.")
    st.download_button(
        "Download configuration",
        json.dumps(current_config(), indent=2).encode("utf-8"),
        file_name="chaos_lab_configuration.json",
        mime="application/json",
        width="stretch",
    )
    if st.button("Reset configuration", width="stretch"):
        for key, value in defaults().items():
            st.session_state[key] = value
        for result_key in ("drive_result", "kapitza_result", "trajectory_result", "mass_result", "lyap_result", "lyap_convergence_result", "flip_result", "validation_result"):
            st.session_state[result_key] = None
        st.rerun()
    st.caption(f"Platform version: {APP_VERSION}")

st.caption("Workspace order: Setup → Run → Results → Analysis → Verification / Export.")

overview_tab, driven_tab, kapitza_tab, double_tab, lyap_tab, flip_tab, validation_tab = st.tabs(
    ["1 · Overview", "2 · Run — Driven pendulum", "3 · Run — Kapitza stability", "4 · Results — Double pendulum", "5 · Analysis — Lyapunov", "6 · Analysis — Flip map", "7 · Verification"]
)

with overview_tab:
    st.subheader("What this platform tests")
    st.write(
        "It tests nonlinear physical behavior and whether that behavior survives numerical checks. "
        "A complicated graph is not automatically evidence of chaos: timestep error, transient data, "
        "degenerate initial conditions, and incorrect sampling can create misleading patterns."
    )
    a, b, c, d = st.columns(4)
    a.metric("Evolution solver", "RK4")
    b.metric("Adaptive reference", "DOP853")
    c.metric("Chaos diagnostic", "Benettin λ")
    d.metric("Analytical precision", "mpmath · 80 dps")
    st.markdown(
        """
        <div class="note"><b>Axis rule:</b> every parameter scan places the scanned parameter on the
        horizontal axis. Time appears on the horizontal axis only for trajectory evolution, and term-like
        diagnostic indices are used only in dedicated single-case analyses.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Corrections to the notebook:**

        - driven and vibrating-pivot plots use one sample per drive period;
        - the Kapitza scan includes damping before describing a late-time response;
        - equal-angle double-pendulum initial conditions are no longer the default;
        - the mass plot is correctly called a response spectrum, not proof of bifurcation;
        - Lyapunov separation is periodically renormalized;
        - long conservative trajectories report energy drift and timestep convergence.
        """
    )

with driven_tab:
    st.subheader("Driven damped pendulum — Poincaré amplitude scan")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Minimum drive amplitude", min_value=0.0, key="drive_a_min")
    c2.number_input("Maximum drive amplitude", min_value=0.01, key="drive_a_max")
    c3.slider("Amplitude scan points", 10, 200, key="drive_a_points")
    c1, c2, c3 = st.columns(3)
    c1.slider("Total drive periods", 100, 800, step=20, key="drive_periods")
    c2.slider("Discarded transient periods", 20, 600, step=20, key="drive_discard")
    c3.slider("RK4 steps per period", 40, 240, step=20, key="drive_steps_per_period")
    invalid = st.session_state.drive_a_min >= st.session_state.drive_a_max or st.session_state.drive_discard >= st.session_state.drive_periods
    if invalid:
        st.error("Maximum amplitude must exceed minimum amplitude, and discarded periods must be fewer than total periods.")
    if st.button("Run driven-pendulum scan", type="primary", disabled=invalid):
        bar, update = progress_reporter("Stroboscopic integration")
        amplitudes = np.linspace(st.session_state.drive_a_min, st.session_state.drive_a_max, st.session_state.drive_a_points)
        st.session_state.drive_result = driven_poincare_scan(
            amplitudes,
            periods=st.session_state.drive_periods,
            discard_periods=st.session_state.drive_discard,
            steps_per_period=st.session_state.drive_steps_per_period,
            progress_callback=update,
        )
        bar.progress(1.0, text="Driven-pendulum scan complete")
    if st.session_state.drive_result is not None:
        result = st.session_state.drive_result
        fig = go.Figure(go.Scattergl(x=result["drive_amplitude"], y=result["theta"], mode="markers", marker={"size":2,"color":result["omega"],"colorscale":"Turbo","opacity":0.5,"colorbar":{"title":"ω"}}))
        fig.update_layout(template="plotly_white", height=500, title="Stroboscopic response versus drive amplitude", xaxis_title="Drive amplitude A — scanned independent variable", yaxis_title="θ sampled once per drive period (rad)")
        st.plotly_chart(fig, width="stretch")
        points_per_amplitude = int(len(result["theta"]) / max(st.session_state.drive_a_points, 1))
        key_results([
            ("Drive amplitudes scanned", st.session_state.drive_a_points, "count"),
            ("Stroboscopic samples per amplitude", points_per_amplitude, "samples"),
            ("Observed θ range", f"{np.min(result['theta']):.10g} to {np.max(result['theta']):.10g}", "rad"),
            ("Drive period", f"{float(result['period'][0]):.10g}", "s"),
        ])
        complete_data_panel("driven", lambda: pd.DataFrame(result), "driven_poincare_scan.csv")

with kapitza_tab:
    st.subheader("Vibrating-pivot inverted pendulum — stabilization scan")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Maximum pivot amplitude (m)", min_value=0.01, max_value=2.0, key="kapitza_a_max")
    c2.slider("Amplitude scan points", 20, 250, key="kapitza_points")
    c3.number_input("Drive frequency Ω (rad/s)", min_value=1.0, max_value=200.0, key="kapitza_frequency")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Damping β (s⁻¹)", min_value=0.0, max_value=5.0, key="kapitza_damping")
    c2.slider("Total periods", 80, 500, step=20, key="kapitza_periods")
    c3.slider("Discarded periods", 20, 400, step=10, key="kapitza_discard")
    threshold = kapitza_critical_amplitude(st.session_state.gravity, st.session_state.l1, st.session_state.kapitza_frequency)
    st.metric("High-frequency approximate threshold", f"a₍crit₎ ≈ {threshold:.5f} m")
    kapitza_invalid = st.session_state.kapitza_discard >= st.session_state.kapitza_periods
    if st.button("Run Kapitza scan", type="primary", disabled=kapitza_invalid):
        bar, update = progress_reporter("Kapitza amplitude scan")
        amplitudes = np.linspace(0.0, st.session_state.kapitza_a_max, st.session_state.kapitza_points)
        st.session_state.kapitza_result = kapitza_stability_scan(
            amplitudes,
            gravity=st.session_state.gravity,
            length=st.session_state.l1,
            drive_frequency=st.session_state.kapitza_frequency,
            damping=st.session_state.kapitza_damping,
            periods=st.session_state.kapitza_periods,
            discard_periods=st.session_state.kapitza_discard,
            progress_callback=update,
        )
        bar.progress(1.0, text="Kapitza scan complete")
    if st.session_state.kapitza_result is not None:
        result = st.session_state.kapitza_result
        fig = go.Figure(go.Scattergl(x=result["pivot_amplitude"], y=result["theta"], mode="markers", marker={"size":2,"color":"#7b2cbf","opacity":0.35}))
        fig.add_vline(x=float(result["critical_amplitude"][0]), line_dash="dash", line_color="#e76f51", annotation_text="high-frequency threshold")
        fig.update_layout(template="plotly_white", height=500, title="Late-time stroboscopic angle versus vibration amplitude", xaxis_title="Pivot amplitude a (m) — scanned independent variable", yaxis_title="θ sampled once per drive period (rad)")
        st.plotly_chart(fig, width="stretch")
        above = result["pivot_amplitude"] >= float(result["critical_amplitude"][0])
        median_dev = float(np.median(result["upright_deviation"][above])) if np.any(above) else np.nan
        key_results([
            ("High-frequency threshold", f"{float(result['critical_amplitude'][0]):.10g}", "m"),
            ("Amplitudes scanned", st.session_state.kapitza_points, "count"),
            ("Median upright deviation above threshold", f"{median_dev:.10g}", "rad"),
            ("Late-time samples", len(result["theta"]), "samples"),
        ])
        complete_data_panel("kapitza", lambda: pd.DataFrame(result), "kapitza_stability_scan.csv")
        st.caption("The analytical threshold is a high-frequency approximation, not an exact boundary for every damping, frequency, and initial condition.")

with double_tab:
    st.subheader("Double-pendulum trajectory and mass response")
    trajectory_panel, mass_panel = st.tabs(["Results — Single trajectory", "Analysis — Upper-mass response"])
    with trajectory_panel:
        c1, c2 = st.columns(2)
        c1.number_input("Duration (s)", min_value=0.1, max_value=200.0, key="trajectory_duration")
        c2.number_input("Timestep dt (s)", min_value=0.0005, max_value=0.1, step=0.001, format="%.4f", key="trajectory_dt")
        if st.button("Run double-pendulum trajectory", type="primary"):
            bar, update = progress_reporter("Double-pendulum trajectory")
            times, states = simulate_double_pendulum(initial_state(), double_params(), duration=st.session_state.trajectory_duration, dt=st.session_state.trajectory_dt, progress_callback=update)
            st.session_state.trajectory_result = {"time":times,"states":states}
            bar.progress(1.0, text="Trajectory complete")
        if st.session_state.trajectory_result is not None:
            result = st.session_state.trajectory_result
            times, states = result["time"], result["states"]
            energy = double_pendulum_energy(states, double_params())
            coordinates = double_pendulum_cartesian(states, double_params())
            relative_drift = np.max(np.abs(energy-energy[0])) / max(abs(energy[0]), np.finfo(float).tiny)
            angle_fig = go.Figure()
            angle_fig.add_scatter(x=times,y=states[:,0],name="θ₁")
            angle_fig.add_scatter(x=times,y=states[:,2],name="θ₂")
            angle_fig.update_layout(template="plotly_white",height=420,title="Angle evolution",xaxis_title="Time (s) — evolution variable",yaxis_title="Angle (rad)")
            st.plotly_chart(angle_fig,width="stretch")
            left,right=st.columns(2)
            path_fig=go.Figure(go.Scattergl(x=coordinates["x2"],y=coordinates["y2"],mode="lines",line={"color":"#007c83","width":1}))
            path_fig.update_layout(template="plotly_white",height=460,title="Physical path of the lower bob",xaxis_title="x₂ (m)",yaxis_title="y₂ (m)",yaxis={"scaleanchor":"x","scaleratio":1})
            left.plotly_chart(path_fig,width="stretch")
            space_time=go.Figure(go.Scatter3d(x=coordinates["x2"],y=coordinates["y2"],z=times,mode="lines",line={"color":times,"colorscale":"Viridis","width":3}))
            space_time.update_layout(template="plotly_white",height=460,title="3D space–time trajectory (x₂, y₂, t)",scene={"xaxis_title":"x₂ (m)","yaxis_title":"y₂ (m)","zaxis_title":"time (s)"})
            right.plotly_chart(space_time,width="stretch")
            st.caption("The pendulum moves in a physical 2D plane. The third coordinate above is time, so this is a space–time visualization rather than invented out-of-plane motion.")
            drift=float(np.max(np.abs(energy-energy[0]))/max(abs(float(energy[0])),np.finfo(float).tiny))
            key_results([
                ("Final time", f"{times[-1]:.10g}", "s"),
                ("Actual uniform dt", f"{float(times[1]-times[0]):.10g}" if len(times)>1 else "n/a", "s"),
                ("Stored states", len(times), "samples"),
                ("Maximum relative energy drift", f"{drift:.10g}", "dimensionless"),
                ("Lower-bob maximum radius", f"{np.max(np.hypot(coordinates['x2'],coordinates['y2'])):.10g}", "m"),
            ])
            complete_data_panel(
                "trajectory",
                lambda: pd.DataFrame({"time":times,"theta1":states[:,0],"omega1":states[:,1],"theta2":states[:,2],"omega2":states[:,3],"energy":energy,**coordinates}),
                "double_pendulum_trajectory.csv",
            )
    with mass_panel:
        st.info("This is a conservative-system peak-response spectrum. It is not automatically a bifurcation diagram because the system has no attracting steady state.")
        c1,c2,c3=st.columns(3)
        c1.number_input("Minimum m₁ (kg)",min_value=0.05,key="mass_min")
        c2.number_input("Maximum m₁ (kg)",min_value=0.1,key="mass_max")
        c3.slider("Mass scan points",10,100,key="mass_points")
        c1,c2=st.columns(2)
        c1.number_input("Duration per mass (s)",min_value=2.0,max_value=60.0,key="mass_duration")
        c2.number_input("Mass-scan dt (s)",min_value=0.002,max_value=0.05,step=0.001,format="%.3f",key="mass_dt")
        mass_estimated=int(st.session_state.mass_points*np.ceil(st.session_state.mass_duration/st.session_state.mass_dt))
        st.caption(f"Estimated trajectory steps across all masses: {mass_estimated:,}; limit: 5,000,000.")
        mass_invalid=st.session_state.mass_min>=st.session_state.mass_max or mass_estimated>5_000_000
        if st.button("Run upper-mass response scan",type="primary",disabled=mass_invalid):
            bar,update=progress_reporter("Mass response scan")
            masses=np.linspace(st.session_state.mass_min,st.session_state.mass_max,st.session_state.mass_points)
            st.session_state.mass_result=double_pendulum_mass_response(masses,double_params(),initial_state=initial_state(),duration=st.session_state.mass_duration,dt=st.session_state.mass_dt,progress_callback=update)
            bar.progress(1.0,text="Mass response scan complete")
        if st.session_state.mass_result is not None:
            result=st.session_state.mass_result
            fig=go.Figure(go.Scattergl(x=result["mass1"],y=result["theta1_peak"],mode="markers",marker={"size":3,"color":"#146c94","opacity":0.5}))
            fig.update_layout(template="plotly_white",height=500,title="Post-transient θ₁ maxima versus upper mass",xaxis_title="Upper mass m₁ (kg) — scanned independent variable",yaxis_title="Local maxima of θ₁ (rad)")
            st.plotly_chart(fig,width="stretch")
            key_results([
                ("Mass values scanned", len(result["scan_mass1"]), "count"),
                ("Detected post-transient maxima", int(np.sum(result["peak_count"])), "peaks"),
                ("Median peaks per mass", f"{float(np.median(result['peak_count'])):.10g}", "peaks"),
                ("Peak-angle range", f"{np.min(result['theta1_peak']):.10g} to {np.max(result['theta1_peak']):.10g}" if len(result['theta1_peak']) else "none", "rad"),
            ])
            complete_data_panel(
                "mass",
                lambda: pd.DataFrame({"mass1":result["mass1"],"theta1_peak":result["theta1_peak"]}),
                "mass_response_scan.csv",
            )

with lyap_tab:
    st.subheader("Largest finite-time Lyapunov exponent")
    c1,c2=st.columns(2)
    c1.number_input("Lyapunov duration (s)",min_value=2.0,max_value=200.0,key="lyap_duration")
    c2.number_input("Lyapunov dt (s)",min_value=0.0005,max_value=0.03,step=0.001,format="%.4f",key="lyap_dt")
    if st.button("Run Benettin Lyapunov analysis",type="primary"):
        bar,update=progress_reporter("Lyapunov renormalization")
        st.session_state.lyap_result=lyapunov_benettin(initial_state(),double_params(),duration=st.session_state.lyap_duration,dt=st.session_state.lyap_dt,progress_callback=update)
        bar.progress(1.0,text="Lyapunov analysis complete")
    if st.session_state.lyap_result is not None:
        result=st.session_state.lyap_result
        estimate=float(result["estimate"])
        fig=go.Figure()
        fig.add_scatter(x=result["time"],y=result["cumulative_exponent"],name="Cumulative λ")
        fig.add_scatter(x=result["time"],y=result["local_exponent"],name="Local interval λ",opacity=.35)
        fig.add_hline(y=0,line_dash="dash",line_color="gray")
        fig.update_layout(template="plotly_white",height=470,title="Renormalized divergence estimate",xaxis_title="Time (s) — analysis progression",yaxis_title="Exponent estimate (s⁻¹)")
        st.plotly_chart(fig,width="stretch")
        if estimate>0:
            st.warning("A positive finite-time estimate suggests sensitive dependence for this trajectory and time window; it is not by itself a universal proof for every initial condition.")
        key_results([
            ("Finite-time λ estimate", f"{estimate:.10g}", "s⁻¹"),
            ("Actual integration dt", f"{float(result['actual_dt']):.10g}", "s"),
            ("Renormalization samples", len(result["time"]), "count"),
            ("Final analysis time", f"{float(result['time'][-1]):.10g}" if len(result['time']) else "n/a", "s"),
        ])
        complete_data_panel(
            "lyapunov",
            lambda: pd.DataFrame({"time":result["time"],"local_exponent":result["local_exponent"],"cumulative_exponent":result["cumulative_exponent"]}),
            "lyapunov_analysis.csv",
        )
        if st.button("Run Lyapunov timestep sensitivity", key="run_lyap_dt_scan"):
            bar,update=progress_reporter("Lyapunov timestep sensitivity")
            dt_values=np.asarray([st.session_state.lyap_dt, st.session_state.lyap_dt/2.0, st.session_state.lyap_dt/4.0])
            st.session_state.lyap_convergence_result=lyapunov_convergence_scan(
                dt_values, initial_state(), double_params(), duration=st.session_state.lyap_duration, progress_callback=update
            )
            bar.progress(1.0,text="Lyapunov timestep sensitivity complete")
        if st.session_state.lyap_convergence_result is not None:
            conv=st.session_state.lyap_convergence_result
            cfig=go.Figure(go.Scatter(x=conv["dt"],y=conv["estimate"],mode="lines+markers",name="λ(dt)"))
            cfig.update_layout(template="plotly_white",height=420,title="Lyapunov estimate versus timestep",xaxis_type="log",xaxis_autorange="reversed",xaxis_title="dt (s) — numerical resolution",yaxis_title="λ estimate (s⁻¹)")
            st.plotly_chart(cfig,width="stretch")
            key_results([
                ("Finest-dt estimate", f"{float(conv['finest_dt_estimate']):.10g}", "s⁻¹"),
                ("Estimate spread", f"{float(conv['absolute_spread']):.10g}", "s⁻¹"),
                ("Relative spread", f"{float(conv['relative_spread']):.10g}", "dimensionless"),
            ])
            complete_data_panel("lyapunov_dt", lambda: pd.DataFrame({"requested_dt":conv["requested_dt"],"actual_dt":conv["actual_dt"],"estimate":conv["estimate"]}), "lyapunov_timestep_sensitivity.csv")

with flip_tab:
    st.subheader("Double-pendulum time-to-flip map")
    c1,c2,c3=st.columns(3)
    c1.slider("Grid resolution per angle",20,120,step=10,key="flip_resolution")
    c2.number_input("Maximum time",min_value=5.0,max_value=150.0,key="flip_tmax")
    c3.number_input("Flip-map dt",min_value=0.005,max_value=0.05,step=0.005,format="%.3f",key="flip_dt")
    estimated=int(st.session_state.flip_resolution**2*np.ceil(st.session_state.flip_tmax/st.session_state.flip_dt))
    st.caption(f"Estimated system-steps: {estimated:,}. The platform blocks requests above 75,000,000.")
    if st.button("Run flip map",type="primary",disabled=estimated>75_000_000):
        bar,update=progress_reporter("Flip-map integration")
        axis=np.linspace(-np.pi,np.pi,st.session_state.flip_resolution)
        st.session_state.flip_result=flip_time_map(axis,axis,double_params(),t_max=st.session_state.flip_tmax,dt=st.session_state.flip_dt,progress_callback=update)
        bar.progress(1.0,text="Flip map complete")
    if st.session_state.flip_result is not None:
        result=st.session_state.flip_result
        result_tmax=float(result["t_max"])
        display=np.where(np.isfinite(result["flip_time"]),result["flip_time"],result_tmax)
        fig=go.Figure(go.Heatmap(x=result["theta1"],y=result["theta2"],z=display.T,colorscale="Turbo",colorbar={"title":"First flip time"}))
        fig.update_layout(template="plotly_white",height=650,title="First-flip time across initial-angle space",xaxis_title="Initial θ₁ (rad) — scanned axis 1",yaxis_title="Initial θ₂ (rad) — scanned axis 2")
        st.plotly_chart(fig,width="stretch")
        st.caption("Cells shown at the maximum time did not flip within the selected observation window. Refine dt and resolution before interpreting fine boundaries as physical structure.")
        finite=result["flip_time"][np.isfinite(result["flip_time"])]
        no_flip_fraction=float(np.mean(~np.isfinite(result["flip_time"])))
        key_results([
            ("Grid resolution", f"{len(result['theta1'])} × {len(result['theta2'])}", "initial-condition cells"),
            ("No-flip fraction", f"{no_flip_fraction:.10g}", "within selected t_max"),
            ("Earliest detected flip", f"{float(np.min(finite)):.10g}" if finite.size else "none", "s"),
            ("Observation window", f"{result_tmax:.10g}", "s"),
            ("Actual integration dt", f"{float(result['actual_dt']):.10g}", "s"),
        ])
        def flip_frame():
            rows=[]
            for i,t1 in enumerate(result["theta1"]):
                for j,t2 in enumerate(result["theta2"]):
                    rows.append((t1,t2,result["flip_time"][i,j]))
            return pd.DataFrame(rows,columns=["theta1_initial","theta2_initial","first_flip_time"])
        complete_data_panel("flip", flip_frame, "double_pendulum_flip_map.csv")

with validation_tab:
    st.subheader("Physical and numerical compliance")
    st.markdown(
        """
        - positive masses, lengths, gravity, timestep, and duration are enforced;
        - conservative double-pendulum runs report mechanical-energy drift;
        - RK4 is compared at `dt` and `dt/2`;
        - an adaptive DOP853 result supplies an independent solver comparison;
        - parameter scans keep the scanned quantity on their plotted axis;
        - drive-dependent diagrams sample at the drive period after discarding transients;
        - expensive flip maps are protected by a system-step limit.
        """
    )
    if st.button("Run compliance checks",type="primary"):
        params=double_params()
        state=initial_state()
        diagnostic=step_convergence_diagnostic(state,params,duration=4.0,dt=0.01)
        _,fixed=simulate_double_pendulum(state,params,duration=4.0,dt=0.005)
        _,adaptive=simulate_double_pendulum_adaptive(state,params,duration=4.0,samples=fixed.shape[0])
        difference=fixed[-1]-adaptive[-1]
        difference[[0,2]]=(difference[[0,2]]+np.pi)%(2*np.pi)-np.pi
        diagnostic["fixed_vs_dop853_final_difference"]=float(np.linalg.norm(difference))
        diagnostic["passed"]=bool(diagnostic["energy_drift_improved"] and diagnostic["fine_relative_energy_drift"]<1e-5 and diagnostic["fixed_vs_dop853_final_difference"]<1e-4)
        st.session_state.validation_result=diagnostic
    if st.session_state.validation_result is not None:
        diagnostic=st.session_state.validation_result
        if diagnostic["passed"]:
            st.success("Built-in numerical compliance checks passed for the current physical parameters and default validation trajectory.")
        else:
            st.error("One or more compliance checks failed. Reduce the timestep or inspect the selected parameters.")
        key_results([
            ("Fine-step relative energy drift", f"{float(diagnostic['fine_relative_energy_drift']):.10g}", "dimensionless"),
            ("Fixed vs DOP853 final difference", f"{float(diagnostic['fixed_vs_dop853_final_difference']):.10g}", "scaled state norm"),
            ("Compliance status", "PASS" if diagnostic["passed"] else "FAIL", "built-in checks"),
        ])
        complete_data_panel("validation", lambda: pd.DataFrame([diagnostic]), "validation_results.csv")
        st.warning("A passing numerical check does not prove that every trajectory or every long-time fractal boundary is exact. Chaotic systems require convergence checks at the specific parameters being interpreted.")
