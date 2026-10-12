# CloudGuard AI
Cloud monitoring anomaly detection using machine learning and MLflow.
Data: 5 labelled ingress-rate time series (79,200 rows). Stack: pandas, scikit-learn, MLflow, Streamlit, Jenkins, Jira.

## Run (from project root, Windows)
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements-dev.txt
    python src/data_validation.py     # checks data/processed/ingress_cleaned.parquet
    python src/train.py               # 15 MLflow runs, registers best model
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    python -m pytest tests -v
    streamlit run app/streamlit_app.py

## Branches
main (deployable) | data-priya (data + EDA) | ml-mlflow | app-streamlit | ci-testing
