"""
Funciones de inferencia para los modelos entrenados.
Expone una API simple de alto nivel que puede ser consumida por Streamlit,
FastAPI u otros sistemas, con soporte para múltiples modelos.
"""
import json

import joblib
import pandas as pd
import yaml

with open("config/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def _load_model(kind, model_name=None):
    with open("models/model_catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if kind == "risk":
        key = "risk_models"
        default = catalog["risk_default"]
    else:
        key = "fraud_models"
        default = catalog["fraud_default"]

    selected = model_name or default
    if selected not in catalog[key]:
        raise ValueError(f"Modelo '{selected}' no encontrado. Disponibles: {list(catalog[key].keys())}")

    return joblib.load(catalog[key][selected]["artifact"])


def _list_models(kind):
    with open("models/model_catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return list(catalog[f"{kind}_models"].keys())


def predict_risk(client_id: str, model_name: str = None) -> dict:
    clients = pd.read_csv("data/clients.csv")
    client = clients[clients["client_id"] == client_id]
    if client.empty:
        return {"error": "Cliente no encontrado"}

    X = client[CFG["models"]["risk"]["features"]]
    model = _load_model("risk", model_name)
    proba = float(model.predict_proba(X)[:, 1][0])

    return {
        "client_id": client_id,
        "model": model_name or "Riesgo - Random Forest",
        "risk_probability": round(proba, 4),
        "risk_label": "ALTO" if proba >= CFG["models"]["risk"]["threshold"] else "BAJO",
    }


def predict_fraud(transaction_id: str, model_name: str = None) -> dict:
    tx = pd.read_csv("data/transactions_scored.csv")
    row = tx[tx["transaction_id"] == transaction_id]
    if row.empty:
        return {"error": "Transacción no encontrada"}

    X = pd.read_csv("data/transactions.csv")
    X = X[X["transaction_id"] == transaction_id][CFG["models"]["fraud"]["features"]].fillna(0)
    model = _load_model("fraud", model_name)

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
        return {
            "transaction_id": transaction_id,
            "client_id": row["client_id"].values[0],
            "model": model_name or "Fraude - Isolation Forest",
            "amount": float(row["amount"].values[0]),
            "fraud_predicted": pred,
            "fraud_probability": round(proba, 4),
        }
    else:
        # IsolationForest
        pred = int((model.predict(X)[0] == -1).astype(int))
        score = float(-model.decision_function(X)[0])
        return {
            "transaction_id": transaction_id,
            "client_id": row["client_id"].values[0],
            "model": model_name or "Fraude - Isolation Forest",
            "amount": float(row["amount"].values[0]),
            "fraud_predicted": pred,
            "anomaly_score": round(score, 4),
        }


def list_top_risk_clients(n=20, model_name=None):
    clients = pd.read_csv("data/clients.csv")
    X = clients[CFG["models"]["risk"]["features"]]
    model = _load_model("risk", model_name)
    clients["risk_probability"] = model.predict_proba(X)[:, 1]
    clients["model"] = model_name or "Riesgo - Random Forest"
    return clients.sort_values("risk_probability", ascending=False).head(n)


def list_top_fraud_transactions(n=20, model_name=None):
    tx = pd.read_csv("data/transactions_scored.csv").copy()
    X = pd.read_csv("data/transactions.csv")[CFG["models"]["fraud"]["features"]].fillna(0)
    model = _load_model("fraud", model_name)

    if hasattr(model, "predict_proba"):
        tx["fraud_predicted"] = model.predict(X)
        tx["fraud_probability"] = model.predict_proba(X)[:, 1]
    else:
        tx["anomaly_score"] = -model.decision_function(X)
        tx["fraud_predicted"] = (model.predict(X) == -1).astype(int)
        tx["fraud_probability"] = None

    tx["model"] = model_name or "Fraude - Isolation Forest"
    return tx.sort_values("anomaly_score", ascending=False).head(n)


def list_risk_models():
    return _list_models("risk")


def list_fraud_models():
    return _list_models("fraud")


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

        # Predicciones con todos los modelos de riesgo
        risk_predictions = {m: predict_risk(client_id, model_name=m) for m in list_risk_models()}

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
            "predicciones_riesgo": risk_predictions,
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

        # Predicciones con todos los modelos de fraude
        fraud_predictions = {m: predict_fraud(transaction_id, model_name=m) for m in list_fraud_models()}

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
            "predicciones_fraude": fraud_predictions,
        }
        return context

    return {"error": "Debe proporcionar client_id o transaction_id"}


def _to_native(value):
    if hasattr(value, "item"):
        return value.item()
    return value
