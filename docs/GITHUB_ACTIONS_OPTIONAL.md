# Optional GitHub Actions CI

The repository ships the test workflow as `docs/github-actions/tests.yml.example` instead of an active `.github/workflows` file. This keeps a normal HTTPS push compatible with Personal Access Tokens that do not have the special workflow permission.

To activate CI later, create `.github/workflows/tests.yml` in the GitHub web interface (or with a token that has workflow permission) and copy the example file into it. The workflow installs `requirements-dev.txt` and runs the test suite on the configured Python versions.
