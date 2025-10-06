.PHONY: venv install lint fmt test run suggest dryrun

venv:
	python -m venv .venv

install:
	. .venv/Scripts/activate || . .venv/bin/activate; pip install -r requirements.txt

lint:
	@echo "Tip: puedes agregar ruff/black aquí si quieres."

fmt:
	@echo "Tip: agrega black -q betcomb tests"

test:
	pytest -q

run:
	python -m betcomb.cli --help

suggest:
	python -m betcomb.cli suggest --leagues LALIGA,LIBERTADORES --days 7 --min-total-odds 2.0 --explain yes --provider-stats mock --provider-odds mock --export-format md

dryrun:
	python -m betcomb.cli dryrun --file betcomb/data/samples/fixtures.json --file betcomb/data/samples/odds.json
