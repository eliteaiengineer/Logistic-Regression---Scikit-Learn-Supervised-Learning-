import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def infer_columns(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c != target]
    categorical = [c for c in features if df[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical]
    return numeric, categorical


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])
    return pre
