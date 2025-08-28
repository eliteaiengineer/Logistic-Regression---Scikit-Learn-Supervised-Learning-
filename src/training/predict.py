import argparse, json
import pandas as pd
import joblib


def main() -> int:
    p = argparse.ArgumentParser(description="Predict with a saved model.")
    p.add_argument("--model", required=True)
    p.add_argument("--json", help="JSON array of records", default=None)
    p.add_argument("--infile", help="CSV or JSONL file with records", default=None)
    args = p.parse_args()

    pipe = joblib.load(args.model)

    if args.json:
        X = pd.DataFrame(json.loads(args.json))
    elif args.infile:
        if args.infile.endswith(".jsonl"):
            X = pd.read_json(args.infile, lines=True)
        else:
            X = pd.read_csv(args.infile)
    else:
        raise SystemExit("Provide --json or --infile")

    preds = pipe.predict(X)
    print(json.dumps(preds.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
