"""Tests de la API de predicción."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predict import predict_risk, predict_fraud, build_rich_context


def test_predict_risk_known_client():
    result = predict_risk("CLI00000001")
    assert "risk_probability" in result
    assert 0 <= result["risk_probability"] <= 1


def test_predict_risk_unknown_client():
    result = predict_risk("CLI_INEXISTENTE")
    assert "error" in result


def test_predict_fraud_known_transaction():
    result = predict_fraud("TXN0000000001")
    assert "anomaly_score" in result


def test_predict_fraud_unknown_transaction():
    result = predict_fraud("TXN_INEXISTENTE")
    assert "error" in result


def test_build_rich_context_client():
    ctx = build_rich_context(client_id="CLI00000001")
    assert ctx["tipo"] == "cliente"
    assert "perfil_cliente" in ctx


def test_build_rich_context_error():
    ctx = build_rich_context(client_id="CLI_INEXISTENTE")
    assert "error" in ctx
