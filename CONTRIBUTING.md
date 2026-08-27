# Contributing

Contributions are welcome when they preserve scientific meaning as well as software correctness.

Before opening a pull request, run the core test suite and explain any new physics assumptions, numerical tolerances, units, or convergence criteria. Parameter scans must use the scanned quantity as the independent axis. Do not label a numerical pattern as a physical transition or chaotic invariant unless the diagnostic actually supports that interpretation.

A typical local check is:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

UI changes should keep the current presentation rule: plots and trends first, a short key-results table second, and complete numerical tables only on request.
