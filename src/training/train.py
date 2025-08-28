import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib

from utils.processing import infer_columns, build_preprocessor
from training.model import build_model


def main() -> int:
    p = argparse.ArgumentParser(description="Train a supervised model (sample).")
    p.add_argument("--data", default="data/sample/customers.csv")
    p.add_argument("--target", default="churn")
    p.add_argument("--model", default="rf", choices=["logreg", "rf"])
    p.add_argument("--out", default="outputs/model.joblib")
    args = p.parse_args()
    print(f"Training {args.model} model on {args.data}, target={args.target}")
    df = pd.read_csv(args.data)
    num, cat = infer_columns(df, args.target)
    
    pre = build_preprocessor(num, cat)
    pipe = build_model(pre, model_type=args.model)
    print("Pipeline steps:", pipe.named_steps)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    print("Data shape:", X.shape, y.shape)
    print("Splitting data...")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("Train shape:", X_tr.shape, y_tr.shape)
    print("Validation shape:", X_va.shape, y_va.shape)
    pipe.fit(X_tr, y_tr)

    preds = pipe.predict(X_va)
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_va, preds))
    pr, rc, f1, _ = precision_recall_fscore_support(y_va, preds, average="binary", zero_division=0)
    metrics.update(dict(precision=float(pr), recall=float(rc), f1=float(f1)))
    try:
        proba = pipe.predict_proba(X_va)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_va, proba))
    except Exception:
        pass

    Path("outputs").mkdir(exist_ok=True)
    joblib.dump(pipe, args.out)
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
