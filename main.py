import os

# Define the structure
paths = [
    "README.md",
    "requirements.txt",
    "Makefile",
    "pytest.ini",
    "data/sample/customers.csv",
    "src/elite_sl/__init__.py",
    "src/elite_sl/processing.py",
    "src/elite_sl/model.py",
    "src/elite_sl/train.py",
    "src/elite_sl/predict.py",
    "tests/test_processing.py",
    "tests/test_train_smoke.py",
]

# Create each path
for path in paths:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        pass  # create empty file

print("✅ Project structure created with empty files.")
