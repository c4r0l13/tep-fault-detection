.PHONY: install data run test lint all

install:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py --out data/raw

run:
	python -m tep_fdd.pipeline --data data/raw --out reports

test:
	pytest -q

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

all: data run test
