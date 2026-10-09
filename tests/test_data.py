"""Tests de calidad y existencia de datos dummy."""
import os

import pandas as pd
import pytest


def test_data_files_exist():
    assert os.path.exists("data/clients.csv")
    assert os.path.exists("data/accounts.csv")
    assert os.path.exists("data/transactions.csv")
    assert os.path.exists("data/transactions_scored.csv")


def test_clients_schema():
    df = pd.read_csv("data/clients.csv")
    expected = [
        "client_id",
        "age",
        "monthly_income",
        "total_debt",
        "credit_score",
        "tenure_months",
        "num_products",
        "client_type",
        "country",
        "segment",
        "debt_to_income",
        "risk_high",
    ]
    for col in expected:
        assert col in df.columns, f"Falta columna {col}"
    assert not df.empty


def test_transactions_schema():
    df = pd.read_csv("data/transactions.csv")
    assert "transaction_id" in df.columns
    assert "client_id" in df.columns
    assert "amount" in df.columns
    assert "fraud_real" in df.columns
    assert not df.empty


def test_risk_target_distribution():
    df = pd.read_csv("data/clients.csv")
    assert set(df["risk_high"].unique()).issubset({0, 1})
