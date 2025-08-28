# 🧠 Supervised Learning (Sample) — Elite AI Engineer

**Goal:** A tiny, production-style supervised learning repo you’ll extend after exercises.  
**Problem:** Predict customer **churn** from mixed numeric + categorical features.  
**Stack:** Python, pandas, scikit-learn (ColumnTransformer + Pipeline), pytest.

## 🚀 Quickstart

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Train a model (RandomForest by default)
make train

# 3) See metrics printed and saved
cat outputs/metrics.json

# 4) Predict on JSON records
make predict
Manual commands (equivalent)
python3 -m src.elite_sl.train \
  --data data/sample/customers.csv \
  --target churn \
  --model rf \
  --out outputs/model.joblib

python3 -m src.elite_sl.predict \
  --model outputs/model.joblib \
  --json '[{"age":29,"income":4200,"city":"Beirut","device":"ios","tenure":4,"is_active":1}]'
📦 What’s inside
Data Processing: median impute + StandardScaler (numeric), most-freq impute + One-Hot (categorical)
Models: Logistic Regression (logreg) or Random Forest (rf)
Artifacts: outputs/model.joblib, outputs/metrics.json
Tests: shape checks, smoke training test
This is a sample. In the course, you’ll add: better EDA, feature engineering, hyperparam tuning, cross-val, MLflow, etc.

---

## requirements.txt

```txt
pandas>=1.5
scikit-learn>=1.3
joblib>=1.3
pytest>=7
```
Makefile
```
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



pytest.ini
```
[pytest]
pythonpath = src
testpaths = tests

```