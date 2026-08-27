# Architecture

`chaos_lab/core.py` owns equations, integration, scans, convergence, and chaos diagnostics. `app.py` owns Streamlit state, plotting, controls, progress, and exports. Tests call the core directly so the scientific logic is not tied to the UI.

Long-result rendering is deliberately lazy: the figure and key results are shown first; complete DataFrames are constructed only after the user enables the complete-data panel.
