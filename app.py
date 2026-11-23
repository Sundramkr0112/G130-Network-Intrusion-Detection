from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from src.config import FEATURES, TARGET
from src.data import load_dataset
from src.drift import check_drift
from src.explain import feature_importance

st.set_page_config(page_title="G130 Intrusion Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ G130 | Network Intrusion Detection")
st.caption("Explainable and drift-aware machine-learning research prototype")

model_path = Path("artifacts/intrusion_model.joblib")
meta_path = Path("artifacts/model_metadata.json")

if not model_path.exists():
    st.warning("No trained model found. Run: python -m src.train --data data/sample_network_flows.csv")
    st.stop()

model = joblib.load(model_path)
metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predict traffic", "Explainability", "Drift monitoring"])

with tab1:
    st.subheader("Model evaluation")
    metrics = metadata.get("metrics", {})
    cols = st.columns(4)
    for col, key, label in zip(cols, ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                               ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]):
        col.metric(label, f"{metrics.get(key, 0):.3f}" if metrics else "Train model")
    if metrics.get("confusion_matrix"):
        cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
        st.write("Confusion matrix")
        st.dataframe(cm, use_container_width=True)
    st.info("Use benchmark datasets for final reported results. The bundled dataset is synthetic demo data.")

with tab2:
    st.subheader("Upload network-flow CSV")
    uploaded = st.file_uploader("CSV must contain the 10 model features", type=["csv"])
    if uploaded:
        try:
            frame = pd.read_csv(uploaded)
            missing = [f for f in FEATURES if f not in frame.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                preds = model.predict(frame[FEATURES])
                result = frame.copy()
                result["Prediction"] = preds
                st.dataframe(result.head(100), use_container_width=True)
                st.download_button("Download predictions", result.to_csv(index=False), "g130_predictions.csv", "text/csv")
                st.metric("Potential intrusion predictions", int(sum(str(p).upper() != "BENIGN" for p in preds)))
        except Exception as exc:
            st.error(str(exc))
    else:
        st.caption("Upload a flow CSV to classify records.")

with tab3:
    st.subheader("Global feature importance")
    importance = feature_importance(model)
    fig = px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h",
                 title="Random Forest feature importance")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This is a global model explanation, not a causal explanation for a specific connection.")

with tab4:
    st.subheader("Reference vs incoming traffic")
    ref_file = st.file_uploader("Reference CSV", type=["csv"], key="ref")
    new_file = st.file_uploader("Incoming CSV", type=["csv"], key="new")
    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
    if st.button("Run drift check"):
        if not ref_file or not new_file:
            st.error("Upload both reference and incoming CSV files.")
        else:
            ref, new = pd.read_csv(ref_file), pd.read_csv(new_file)
            missing = [f for f in FEATURES if f not in ref.columns or f not in new.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                report = check_drift(ref, new, alpha)
                st.metric("Features with drift signal", f"{report['features_flagged']} / {report['features_checked']}")
                st.dataframe(pd.DataFrame(report["results"]), use_container_width=True)
                st.warning(report["interpretation"])
