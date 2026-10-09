"""
Aplicación Streamlit para consumir modelos de riesgo/fraude
y exponer un agente IA avanzado con Mistral y Groq (fallback).
"""
import importlib
import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.llm_agent
import src.predict
import src.drift
importlib.reload(src.llm_agent)
importlib.reload(src.predict)
importlib.reload(src.drift)

from src.llm_agent import ask
from src.retrain import retrain
from src.drift import run_drift_report
from src.predict import (
    build_rich_context,
    list_top_fraud_transactions,
    list_top_risk_clients,
)

st.set_page_config(
    page_title="Agente IA - Riesgo y Fraude",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agente Inteligente de Riesgo y Fraude")
st.markdown("Solución analítica productiva para scoring de riesgo, detección de fraude y asesoría IA.")

tab_risk, tab_fraud, tab_agent, tab_monitor, tab_drift = st.tabs(["Riesgo Financiero", "Fraude", "Agente IA", "Monitoreo", "Drift"])

with tab_risk:
    st.header("Clientes con mayor riesgo financiero")
    top_clients = list_top_risk_clients(20)
    st.dataframe(top_clients, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            top_clients,
            x="risk_probability",
            nbins=20,
            title="Distribución probabilidad de riesgo (top 20)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(
            top_clients.head(10).sort_values("risk_probability"),
            x="client_id",
            y="risk_probability",
            color="risk_probability",
            title="Top 10 clientes de mayor riesgo",
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab_fraud:
    st.header("Transacciones más anómalas")
    top_tx = list_top_fraud_transactions(20)
    st.dataframe(top_tx, use_container_width=True)

    fig3 = px.scatter(
        top_tx,
        x="amount",
        y="anomaly_score",
        color="fraud_predicted",
        hover_data=["transaction_id", "client_id"],
        title="Monto vs Score de anomalía",
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab_agent:
    st.header("Consulta al Agente IA")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.selectbox(
            "Modo del agente",
            ["expert", "executive", "technical", "alert"],
            format_func=lambda x: {
                "expert": "Experto",
                "executive": "Ejecutivo",
                "technical": "Técnico",
                "alert": "Alerta operativa",
            }[x],
        )
    with col2:
        provider = st.selectbox("Proveedor (auto = fallback)", ["auto", "groq", "mistral"])

    with st.expander("Configuración avanzada de modelos"):
        groq_model = st.selectbox(
            "Modelo Groq",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
            index=0,
        )
        mistral_model = st.selectbox(
            "Modelo Mistral",
            ["open-mistral-nemo", "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
            index=0,
        )

    mode_selected = st.radio("Tipo de consulta", ["Cliente", "Transacción"], horizontal=True)

    context = {}
    if mode_selected == "Cliente":
        client_id = st.text_input("Client ID", "CLI00000001")
        if client_id:
            context = build_rich_context(client_id=client_id)
    else:
        tx_id = st.text_input("Transaction ID", "TXN0000000001")
        if tx_id:
            context = build_rich_context(transaction_id=tx_id)

    if context:
        with st.expander("Contexto enriquecido enviado al agente"):
            st.json(context)

    question = st.text_area(
        "Pregunta",
        "¿Qué deberíamos hacer con este cliente/transacción?",
    )

    history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]

    if st.button("Preguntar", type="primary", key="ask_button"):
        if not context or "error" in context:
            st.warning("Ingrese un ID válido antes de consultar.")
        else:
            with st.spinner("Consultando modelo de lenguaje..."):
                raw = ask(
                    context,
                    question,
                    provider=provider,
                    mode=mode,
                    chat_history=history,
                    groq_model=groq_model,
                    mistral_model=mistral_model,
                )

            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": raw})

            st.subheader("Respuesta del agente")

            import re
            urgencia_match = re.search(r"urgencia[:\s]+\n?\s*(alta|media|baja)", raw, re.IGNORECASE)
            riesgo_match = re.search(r"riesgo score[:\s]+\n?\s*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Urgencia", urgencia_match.group(1).strip().upper() if urgencia_match else "No especificada")
            with col_b:
                st.metric("Riesgo score", riesgo_match.group(1).strip() if riesgo_match else "No especificado")

            st.markdown(raw)

    if st.session_state.messages:
        with st.expander("Ver historial completo de la conversación"):
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

with tab_monitor:
    st.header("Métricas y estado de modelos")
    try:
        with open("models/model_registry.json", "r", encoding="utf-8") as f:
            registry = json.load(f)

        st.markdown(
            f"**Proyecto:** {registry['project']} | "
            f"**Versión:** {registry['version']} | "
            f"**Último entrenamiento:** {registry['trained_at'][:10]}"
        )

        risk = registry["models"]["risk"]
        fraud = registry["models"]["fraud"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Modelos activos", len(registry["models"]))
        with col2:
            st.metric("Riesgo ROC-AUC", risk["metrics"].get("roc_auc", "N/A"))
        with col3:
            st.metric("Fraude F1", fraud["metrics"].get("f1", "N/A"))
        with col4:
            st.metric("Transacciones analizadas", 100_000)

        st.subheader("Detalle por modelo")
        rows = []
        for name, m in registry["models"].items():
            rows.append({
                "Modelo": m["name"],
                "Versión": m["version"],
                "Estado": m["status"].upper(),
                "Métricas": ", ".join([f"{k}: {v}" for k, v in m["metrics"].items()]),
                "Entrenado": registry["trained_at"][:10],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Comparativa de métricas")
        metrics_data = []
        for name, m in registry["models"].items():
            for k, v in m["metrics"].items():
                metrics_data.append({"Modelo": m["name"], "Métrica": k, "Valor": v})
        if metrics_data:
            dfm = pd.DataFrame(metrics_data)
            fig = px.bar(
                dfm,
                x="Métrica",
                y="Valor",
                color="Modelo",
                barmode="group",
                title="Métricas de modelos entrenados",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Retraining automático")
        from src.retrain import should_retrain
        if should_retrain(days=30):
            st.warning("Los modelos tienen más de 30 días. Considere reentrenar.")
        else:
            st.info("Los modelos están vigentes.")

        if st.button("Reentrenar modelos ahora", key="retrain_button"):
            with st.spinner("Entrenando modelos, esto puede tardar unos minutos..."):
                result = retrain(force=True)
            st.success(result)

    except FileNotFoundError:
        st.warning("Aún no se han entrenado modelos. Ejecute `python src/train_models.py`")

with tab_drift:
    st.header("Monitoreo de drift")
    st.markdown("Comparación entre datos de referencia (entrenamiento) y datos actuales simulados usando PSI y KS.")

    if st.button("Ejecutar análisis de drift", key="drift_button"):
        with st.spinner("Analizando drift..."):
            report = run_drift_report()

        if report["drift_detected"]:
            st.error(f"Drift detectado en: {report['risk_drifted_features'] + report['fraud_drifted_features']}")
        else:
            st.success("No se detectó drift significativo.")

        st.subheader("Variables de riesgo")
        st.dataframe(pd.DataFrame(report["risk_report"]).T, use_container_width=True)

        st.subheader("Variables de fraude")
        st.dataframe(pd.DataFrame(report["fraud_report"]).T, use_container_width=True)

        if report["drift_detected"]:
            if st.button("Reentrenar con datos actuales", key="retrain_drift_button"):
                with st.spinner("Reentrenando..."):
                    result = retrain(force=True)
                st.success(result)
