"""Tests de carga y comportamiento de modelos entrenados."""
import os

import joblib
import pytest


def test_risk_artifact_exists():
    assert os.path.exists("models/risk_model_v1.0.0.joblib")


def test_fraud_artifact_exists():
    assert os.path.exists("models/fraud_model_v1.0.0.joblib")


def test_risk_model_loads():
    model = joblib.load("models/risk_model_v1.0.0.joblib")
    assert hasattr(model, "predict_proba")


def test_fraud_model_loads():
    model = joblib.load("models/fraud_model_v1.0.0.joblib")
    assert hasattr(model, "predict")
