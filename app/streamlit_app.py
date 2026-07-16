import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.config import settings
from fraud_detection.inference import FraudPredictor

st.set_page_config(page_title="Fraud Sentinel", page_icon="🛡️", layout="wide")
st.title("Fraud Sentinel")
st.caption("Detecção de anomalias e fraudes em transações de cartão")

if not settings.artifact_path.exists():
    st.warning("Treine um modelo antes de usar o painel: `python -m fraud_detection.cli train --demo`.")
    st.stop()

@st.cache_resource(show_spinner="Carregando modelo de fraude...")
def load_predictor() -> FraudPredictor:
    return FraudPredictor()


predictor = load_predictor()
artifact = predictor.artifact
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo", artifact["model_name"])
c2.metric("Features", artifact["feature_count"])
c3.metric("Limiar", f"{artifact['threshold']:.4f}")
c4.metric("Fraudes no treino", artifact["dataset"]["frauds"])

upload = st.file_uploader("Envie um CSV com Time, Amount e V1...V28", type="csv")
if upload:
    data = pd.read_csv(upload)
    try:
        predictions = predictor.predict(data.to_dict(orient="records"))
        output = pd.concat([data.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
        flagged = int(output["fraud"].sum())
        st.metric("Transações sinalizadas", flagged)
        st.dataframe(output.sort_values("score", ascending=False), use_container_width=True)
        st.download_button("Baixar resultados", output.to_csv(index=False), "predictions.csv", "text/csv")
    except ValueError as exc:
        st.error(str(exc))

st.subheader("Comparação dos modelos")
chart = pd.DataFrame(artifact["metrics"]).set_index("name")[["pr_auc", "roc_auc", "recall", "precision"]]
st.bar_chart(chart)
