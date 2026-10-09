"""
Funciones de inferencia para los modelos entrenados.
Expone una API simple de alto nivel que puede ser consumida por Streamlit
u otros sistemas.
"""
import json

import joblib
import pandas as pd
import yaml

with open("config/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def _load_model(kind):
    artifact = CFG["models"][kind]["artifact"]
    return joblib.load(artifact)


def _to_native(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _feature_importance(model, kind):
    if kind == "risk":
        clf = model.named_steps["clf"]
        return dict(zip(CFG["models"]["risk"]["features"], clf.feature_importances_.tolist()))
    return {}


def predict_risk(client_id: str) -> dict:
    clients = pd.read_csv("data/clients.csv")
    client = clients[clients["client_id"] == client_id]
    if client.empty:
        return {"error": "Cliente no encontrado"}

    X = client[CFG["models"]["risk"]["features"]]
    model = _load_model("risk")
    proba = float(model.predict_proba(X)[:, 1][0])
    top_features = _feature_importance(model, "risk")

    return {
        "client_id": client_id,
        "risk_probability": round(proba, 4),
        "risk_label": "ALTO" if proba >= CFG["models"]["risk"]["threshold"] else "BAJO",
        "top_features": {k: round(v, 4) for k, v in top_features.items()},
    }


def predict_fraud(transaction_id: str) -> dict:
    tx = pd.read_csv("data/transactions_scored.csv")
    row = tx[tx["transaction_id"] == transaction_id]
    if row.empty:
        return {"error": "Transacción no encontrada"}

    return {
        "transaction_id": transaction_id,
        "client_id": row["client_id"].values[0],
        "amount": float(row["amount"].values[0]),
        "anomaly_score": float(row["anomaly_score"].values[0]),
        "fraud_predicted": int(row["fraud_predicted"].values[0]),
        "fraud_real": int(row["fraud_real"].values[0]),
        "channel": row["channel"].values[0],
        "hour": int(row["hour"].values[0]),
    }


def list_top_risk_clients(n=20):
    clients = pd.read_csv("data/clients.csv")
    X = clients[CFG["models"]["risk"]["features"]]
    model = _load_model("risk")
    clients["risk_probability"] = model.predict_proba(X)[:, 1]
    return clients.sort_values("risk_probability", ascending=False).head(n)


def list_top_fraud_transactions(n=20):
    tx = pd.read_csv("data/transactions_scored.csv")
    return tx.sort_values("anomaly_score", ascending=False).head(n)


def build_rich_context(client_id: str = None, transaction_id: str = None) -> dict:
    """Construye un contexto enriquecido para el agente IA."""
    if client_id:
        clients = pd.read_csv("data/clients.csv")
        client = clients[clients["client_id"] == client_id]
        if client.empty:
            return {"error": "Cliente no encontrado"}

        client_row = client.iloc[0].to_dict()
        client_row = {k: _to_native(v) for k, v in client_row.items()}

        accounts = pd.read_csv("data/accounts.csv")
        client_accounts = accounts[accounts["client_id"] == client_id]

        tx = pd.read_csv("data/transactions_scored.csv")
        client_tx = tx[tx["client_id"] == client_id]

        recent_tx = client_tx.sort_values("timestamp", ascending=False).head(5)
        recent_tx = recent_tx.fillna(0).to_dict(orient="records")

        risk_pred = predict_risk(client_id)

        context = {
            "tipo": "cliente",
            "perfil_cliente": client_row,
            "cuentas": client_accounts.fillna(0).to_dict(orient="records"),
            "resumen_transaccional": {
                "total_transacciones": int(len(client_tx)),
                "monto_promedio": round(float(client_tx["amount"].mean()), 2) if not client_tx.empty else 0,
                "monto_maximo": int(client_tx["amount"].max()) if not client_tx.empty else 0,
                "transacciones_fraudulentas_detectadas": int(client_tx["fraud_predicted"].sum()) if not client_tx.empty else 0,
            },
            "transacciones_recientes": recent_tx,
            "prediccion_riesgo": risk_pred,
        }
        return context

    if transaction_id:
        tx = pd.read_csv("data/transactions_scored.csv")
        row = tx[tx["transaction_id"] == transaction_id]
        if row.empty:
            return {"error": "Transacción no encontrada"}

        tx_row = row.iloc[0].to_dict()
        tx_row = {k: _to_native(v) for k, v in tx_row.items()}
        client_id = tx_row["client_id"]
        client_ctx = build_rich_context(client_id=client_id)

        # transacciones similares: mismo canal y rango de monto
        similar = tx[
            (tx["client_id"] == client_id)
            & (tx["channel"] == tx_row["channel"])
            & (tx["amount"].between(tx_row["amount"] * 0.8, tx_row["amount"] * 1.2))
        ].head(5)

        context = {
            "tipo": "transaccion",
            "transaccion_actual": tx_row,
            "contexto_cliente": client_ctx,
            "transacciones_similares": similar.fillna(0).to_dict(orient="records"),
            "prediccion_fraude": predict_fraud(transaction_id),
        }
        return context

    return {"error": "Debe proporcionar client_id o transaction_id"}
