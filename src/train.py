"""Train 3 anomaly detectors on data/processed/ingress_cleaned.parquet,
log every run to MLflow and register the best one. Run from project root."""
import os, joblib, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from features import make_features, FEATURES

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("CloudGuard-Anomaly-Detection")
DATA = "data/processed/ingress_cleaned.parquet"

def configs():
    for c in (0.01, 0.02, 0.05):
        for n in (100, 200):
            yield "IsolationForest", {"n_estimators": n, "contamination": c}, IsolationForest(n_estimators=n, contamination=c, random_state=42)
    for k in (20, 50):
        for c in (0.01, 0.02, 0.05):
            yield "LocalOutlierFactor", {"n_neighbors": k, "contamination": c}, LocalOutlierFactor(n_neighbors=k, contamination=c, novelty=True)
    for nu in (0.01, 0.02, 0.05):
        yield "OneClassSVM", {"nu": nu, "kernel": "rbf"}, OneClassSVM(nu=nu, kernel="rbf", gamma="scale")

def main(data=DATA):
    df = make_features(pd.read_parquet(data))
    X, y = df[FEATURES], df["Label"]
    results = []
    for name, params, est in configs():
        Xfit = X.sample(8000, random_state=42) if name == "OneClassSVM" else X   # SVM is O(n^2)
        pipe = make_pipeline(StandardScaler(), est).fit(Xfit)
        pred = (pipe.predict(X) == -1).astype(int)
        m = {"f1": f1_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
             "recall": recall_score(y, pred), "anomalies_found": int(pred.sum())}
        for s, idx in df.groupby("Series").groups.items():
            if y[idx].sum() > 0:
                m["f1_" + s.replace("-", "_")] = f1_score(y[idx], pred[idx])
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"model": name, **params, "train_rows": len(Xfit)})
            mlflow.log_metrics(m)
            mlflow.sklearn.log_model(pipe, "model", serialization_format="cloudpickle")
        results.append({"model": name, **params, **{k: m[k] for k in ("f1", "precision", "recall", "anomalies_found")},
                        "run_id": run.info.run_id, "pipe": pipe})
        print(f"{name:20s} {params} F1={m['f1']:.3f}")
    best = max(results, key=lambda r: r["f1"])
    mlflow.register_model(f"runs:/{best['run_id']}/model", "cloudguard-anomaly-detector")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best["pipe"], "models/best_model.joblib")
    pd.DataFrame([{k: v for k, v in r.items() if k != "pipe"} for r in results]).to_csv("models/comparison.csv", index=False)
    print("BEST:", best["model"], round(best["f1"], 3))

if __name__ == "__main__":
    main()
