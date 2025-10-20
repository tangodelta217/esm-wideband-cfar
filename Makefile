.PHONY: setup test demo eval lint docker-run

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

demo:
	python scripts/demo_cfar.py

eval:
	python eval/evaluate_cfar.py

lint:
	ruff check src scripts tests eval

docker-run:
	docker compose run --rm app
