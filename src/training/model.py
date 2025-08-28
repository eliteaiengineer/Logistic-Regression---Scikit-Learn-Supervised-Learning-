from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_model(preprocessor, model_type: str = "logreg", **kwargs) -> Pipeline:
    if model_type == "rf":
        est = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            random_state=42,
        )
    else:
        est = LogisticRegression(max_iter=1000)
    return Pipeline([("pre", preprocessor), ("clf", est)])
