from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_github_repository_documents_exist() -> None:
    for relative in (
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "RELEASE_CHECKLIST.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/pull_request_template.md",
        "docs/GITHUB_ACTIONS_OPTIONAL.md",
        "docs/github-actions/tests.yml.example",
    ):
        assert (ROOT / relative).is_file(), relative


def test_active_workflow_is_intentionally_not_shipped() -> None:
    assert not (ROOT / ".github/workflows").exists()


def test_launcher_uses_dynamic_port_helper() -> None:
    command_files = list(ROOT.glob("RUN_*.command"))
    assert command_files
    text = command_files[0].read_text()
    assert "tools/find_free_port.py" in text
    assert '--server.port "$PORT"' in text


def test_port_helper_returns_a_port_in_supported_range() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/find_free_port.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 8501 <= int(result.stdout.strip()) <= 8520
