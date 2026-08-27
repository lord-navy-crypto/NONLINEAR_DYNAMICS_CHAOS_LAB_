# Release checklist

- [ ] Version number updated in `app.py`, `pyproject.toml`, and presets
- [ ] `python -m pytest -q tests/test_core.py` passes
- [ ] Python files compile
- [ ] JSON/TOML configuration parses
- [ ] macOS launcher syntax passes and executable bit is preserved
- [ ] Windows launcher still uses the project-local virtual environment
- [ ] README matches actual features and limitations
- [ ] Generated caches and `.venv` are excluded
- [ ] ZIP integrity test passes
- [ ] SHA-256 checksum generated
- [ ] Optional GitHub Actions workflow is activated only when the uploader has workflow permission
