.PHONY: test core-test compile

test:
	python -m pytest -q

core-test:
	python -m pytest -q tests/test_core.py tests/test_repository.py

compile:
	python -m compileall -q app.py */core.py */__init__.py tools
