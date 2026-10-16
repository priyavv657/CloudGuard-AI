import os, sys, joblib, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import make_features, FEATURES
DATA = "data/processed/ingress_cleaned.parquet"

def df(): return pd.read_parquet(DATA)

def test_columns():
    assert {"TimeStamp", "Value", "Label", "Series"} <= set(df().columns)

def test_labels_binary_and_present():
    d = df(); assert set(d.Label.unique()) <= {0, 1} and d.Label.sum() > 0

def test_no_missing_or_duplicates():
    d = df(); assert d.isna().sum().sum() == 0
    assert not d.duplicated(["Series", "TimeStamp"]).any()

def test_features():
    f = make_features(df()); assert f[FEATURES].notna().all().all()

def test_model_predicts():
    f = make_features(df()); p = joblib.load("models/best_model.joblib").predict(f[FEATURES])
    assert set(p) <= {-1, 1} and len(p) == len(f)

def test_comparison_has_three_models():
    assert pd.read_csv("models/comparison.csv").model.nunique() == 3
