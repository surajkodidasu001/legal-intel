.PHONY: install data experiments report api test lint clean

install:
	pip install -r requirements.txt

data:            ## ingest -> dedup -> split, writes data/processed/
	python scripts/prepare_data.py --config configs/milestone1.yaml

experiments:     ## run every dev-split experiment, writes results/*.json
	python scripts/run_experiment.py --config configs/milestone1.yaml

report:          ## regenerate every README table from results/*.json
	python scripts/build_report.py --out README_RESULTS.md

api:
	uvicorn legalintel.api.main:app --reload --app-dir src

test:
	pytest -q

clean:
	rm -rf .cache artifacts/* results/*
