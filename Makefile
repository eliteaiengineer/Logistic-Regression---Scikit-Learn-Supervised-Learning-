install:
	pip install -r requirements.txt

train:
	python3 -m src.training.train \
		--data data/sample/customers.csv \
		--target churn \
		--model rf \
		--out outputs/model.joblib

predict:
	python3 -m src.training.predict \
		--model outputs/model.joblib \
		--json '[{"age":29,"income":4200,"city":"Beirut","device":"ios","tenure":4,"is_active":1}]'

test:
	pytest -v