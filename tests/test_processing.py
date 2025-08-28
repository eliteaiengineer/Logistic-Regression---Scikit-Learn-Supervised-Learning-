import pandas as pd
from utils.processing import infer_columns, build_preprocessor


def test_infer_and_preprocessor():
    df = pd.DataFrame({
        "age": [20, 30, None],
        "city": ["Beirut", "Saida", None],
        "churn": [0, 1, 0]
    })
    num, cat = infer_columns(df, "churn")
    assert num == ["age"]
    assert cat == ["city"]
    pre = build_preprocessor(num, cat)
    Xt = pre.fit_transform(df.drop(columns=["churn"]))
    assert Xt.shape[1] == 3  # 1 numeric + 2 OHE categories
