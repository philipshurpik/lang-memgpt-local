.PHONY: tests lint format evals

evals:
	LANGCHAIN_TEST_CACHE=tests/evals/cassettes poetry run python -m pytest -p no:asyncio  --max-asyncio-tasks 4 tests/evals

lint:
	poetry run ruff check .

format:
	poetry run ruff check --select I --fix
	poetry run ruff format .
	poetry run ruff check . --fix

test:
	poetry run pytest

build:
	poetry build

run:
	poetry run python example_local.py

install:
	poetry install

update:
	poetry updates
