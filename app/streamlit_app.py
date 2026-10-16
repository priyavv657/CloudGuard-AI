import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import joblib, pandas as pd, plotly.graph_objects as go, streamlit as st
from features import make_features, FEATURES

st.set_page_config(page_title="CloudGuard AI", page_icon="☁️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))

@st.cache_data
def load_default():
    return pd.read_parquet(os.path.join(ROOT, "data", "processed", "ingress_cleaned.parquet"))

def detect(df):
    f = make_features(df)
    f["Anomaly"] = load_model().predict(f[FEATURES]) == -1
    return f

page = st.sidebar.radio("CloudGuard AI", ["Dashboard", "Upload / Select Data", "Run Detection",
                                          "Results", "Model Comparison", "About"])
if "data" not in st.session_state:
    st.session_state.data = load_default()
data = st.session_state.data
series_list = sorted(data["Series"].unique())

if page == "Dashboard":
    st.title("☁️ Cloud Monitoring Dashboard")
    res = st.session_state.get("result")
    if res is None:
        st.info("Go to **Run Detection** to analyse the selected series.")
        st.metric("Total data points", f"{len(data):,}")
    else:
        n, a = len(res), int(res.Anomaly.sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total data points", f"{n:,}"); c2.metric("Anomalies detected", a)
        c3.metric("Anomaly %", f"{100*a/n:.2f}%")
        if "Label" in res: c4.metric("Known labelled anomalies", int(res.Label.sum()))
        fig = go.Figure(go.Scatter(x=res.TimeStamp, y=res.Value, name="Ingress rate", line=dict(color="#2f80ed")))
        an = res[res.Anomaly]
        fig.add_trace(go.Scatter(x=an.TimeStamp, y=an.Value, mode="markers", name="Detected anomaly", marker=dict(color="red", size=6)))
        if "Label" in res:
            lb = res[res.Label == 1]
            fig.add_trace(go.Scatter(x=lb.TimeStamp, y=lb.Value, mode="markers", name="Labelled anomaly",
                                     marker=dict(color="orange", size=8, symbol="circle-open")))
        fig.update_layout(title=f"Ingress Rate Over Time ({st.session_state.get('series','')})", height=440)
        st.plotly_chart(fig, width="stretch")

elif page == "Upload / Select Data":
    st.title("Upload / Select Data")
    choice = st.selectbox("Select a built-in series", series_list)
    if st.button("Use this series"):
        st.session_state.series = choice; st.session_state.pop("result", None)
        st.session_state.sel = data[data.Series == choice]; st.success(f"Selected {choice}")
    up = st.file_uploader("...or upload CSV with columns: TimeStamp, Value (optional Label)", type="csv")
    if up:
        d = pd.read_csv(up); d.columns = [c.strip().lower() for c in d.columns]
        if {"timestamp", "value"} <= set(d.columns):
            u = pd.DataFrame({"TimeStamp": pd.to_datetime(d["timestamp"], utc=True, errors="coerce"),
                              "Value": pd.to_numeric(d["value"], errors="coerce"), "Series": "uploaded"})
            if "label" in d: u["Label"] = d["label"].fillna(0).astype(int)
            st.session_state.sel = u.dropna(); st.session_state.series = "uploaded"
            st.session_state.pop("result", None); st.success(f"Loaded {len(u):,} rows")
        else:
            st.error("CSV must contain 'TimeStamp' and 'Value' columns.")
    sel = st.session_state.get("sel", data[data.Series == series_list[0]])
    st.write(f"Preview ({st.session_state.get('series', series_list[0])})"); st.dataframe(sel.head(20))

elif page == "Run Detection":
    st.title("Run Detection")
    sel = st.session_state.get("sel", data[data.Series == series_list[0]])
    st.write(f"Series: **{st.session_state.get('series', series_list[0])}**, {len(sel):,} points")
    if st.button("Run anomaly detection", type="primary"):
        st.session_state.result = detect(sel); st.success("Done. Open Dashboard or Results.")

elif page == "Results":
    st.title("Anomaly Detection Results")
    res = st.session_state.get("result")
    if res is None: st.info("Run detection first.")
    else:
        t = res[res.Anomaly][["TimeStamp", "Value"] + (["Label"] if "Label" in res else [])]
        st.dataframe(t, width="stretch")
        st.download_button("Download anomalies CSV", t.to_csv(index=False), "anomalies.csv")

elif page == "Model Comparison":
    st.title("Model Comparison (MLflow runs)")
    comp = pd.read_csv(os.path.join(ROOT, "models", "comparison.csv"))
    st.subheader("Best F1 per model")
    st.bar_chart(comp.groupby("model")["f1"].max())
    st.dataframe(comp.drop(columns=["run_id"]).sort_values("f1", ascending=False), width="stretch")

else:
    st.title("About")
    st.write("CloudGuard AI detects anomalies in cloud ingress-rate time series using Isolation Forest, "
             "Local Outlier Factor and One-Class SVM, with experiments tracked in MLflow. "
             "Data: 5 labelled ingress-rate series (79,200 observations).")
