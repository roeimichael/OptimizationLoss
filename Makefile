.PHONY: install test clean format lint help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run validation tests"
	@echo "  make example    - Run example usage"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Run code linting"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

test:
	python transductive_saturation_loss.py

example:
	python example_usage.py

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build dist .pytest_cache .mypy_cache

format:
	black *.py

lint:
	flake8 *.py
	mypy *.py
