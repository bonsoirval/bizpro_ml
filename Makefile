.PHONY: run-eda prep train evaluate serve test clean

run-eda:
	python 2_data_understanding/eda.py

prep:
	python 3_data_preparation/prep_pipeline.py

train:
	python 4_modeling/train_model.py

evaluate:
	python 5_evaluation/evaluate.py

serve:
	python 6_deployment/app.py

test:
	pytest -q

clean:
	rm -rf artifacts mlruns *.pyc __pycache__ .pytest_cache
