from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def test_app_loads_all_experiment_sections() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    labels = [tab.label for tab in app.tabs]
    for expected in (
        "Overview",
        "Driven pendulum",
        "Kapitza stability",
        "Double pendulum",
        "Lyapunov",
        "Flip map",
        "Validation",
        "Single trajectory",
        "Upper-mass response scan",
    ):
        assert expected in labels
    assert any(metric.label == "Evolution solver" for metric in app.metric)
    assert any(item.label == "Run double-pendulum trajectory" for item in app.button)


def test_short_trajectory_runs_without_ui_exception() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    _button(app, "Run double-pendulum trajectory").click()
    app.run(timeout=30)
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Maximum relative energy drift" in labels
