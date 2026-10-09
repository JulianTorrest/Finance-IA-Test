"""
API REST con FastAPI para consumir los modelos de riesgo y fraude.
Permite seleccionar el modelo y obtener predicciones estructuradas.
"""
import os
import sys

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.drift import run_drift_report
from src.predict import (
    build_rich_context,
    list_fraud_models,
    list_risk_models,
    predict_fraud,
    predict_risk,
)
from src.retrain import retrain

app = FastAPI(
    title="Finance IA API",
    description="API de modelos de riesgo y fraude con selección de modelos.",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str


class RiskResponse(BaseModel):
    client_id: str
    model: str
    risk_probability: float
    risk_label: str


class FraudResponse(BaseModel):
    transaction_id: str
    client_id: str
    model: str
    amount: float
    fraud_predicted: int
    fraud_probability: Optional[float]


class CatalogResponse(BaseModel):
    risk_models: List[str]
    fraud_models: List[str]


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/catalog", response_model=CatalogResponse)
def catalog():
    return {
        "risk_models": list_risk_models(),
        "fraud_models": list_fraud_models(),
    }


@app.get("/predict/risk/{client_id}", response_model=RiskResponse)
def risk(client_id: str, model: str = None):
    result = predict_risk(client_id, model_name=model)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/predict/fraud/{transaction_id}", response_model=FraudResponse)
def fraud(transaction_id: str, model: str = None):
    result = predict_fraud(transaction_id, model_name=model)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/context/client/{client_id}")
def client_context(client_id: str):
    ctx = build_rich_context(client_id=client_id)
    if "error" in ctx:
        raise HTTPException(status_code=404, detail=ctx["error"])
    return ctx


@app.get("/context/transaction/{transaction_id}")
def transaction_context(transaction_id: str):
    ctx = build_rich_context(transaction_id=transaction_id)
    if "error" in ctx:
        raise HTTPException(status_code=404, detail=ctx["error"])
    return ctx


@app.post("/retrain")
def retrain_models(force: bool = False):
    return {"message": retrain(force=force)}


@app.get("/drift")
def drift_report():
    return run_drift_report()
