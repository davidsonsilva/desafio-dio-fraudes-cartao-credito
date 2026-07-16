install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

train-demo:
	python -m fraud_detection.cli train --demo --rows 5000

api:
	uvicorn fraud_detection.api:app --reload

ui:
	streamlit run app/streamlit_app.py
