"""
Monitoreo de drift para variables de riesgo y fraude.
Usa PSI (Population Stability Index) y test KS.
"""
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


RISK_NUMERIC_COLS = [
    "age",
    "monthly_income",
    "total_debt",
    "credit_score",
    "debt_to_income",
    "num_products",
]

FRAUD_NUMERIC_COLS = [
    "amount",
    "hour",
    "velocity_1h",
    "amount_zscore",
    "country_risk_score",
]


def _psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calcula el Population Stability Index entre dos series."""
    expected = expected.dropna()
    actual = actual.dropna()

    if expected.empty or actual.empty:
        return 0.0

    # Bins basados en la referencia
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 1:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Evita división por cero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    return float(np.sum(psi_values))


def _ks(reference: pd.Series, current: pd.Series) -> float:
    """Calcula la estadística KS de dos muestras."""
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return 0.0
    return float(ks_2samp(ref, cur).statistic)


def _synthetic_current(reference_df: pd.DataFrame, drift_factor: float = 0.15, random_state: int = 2024) -> pd.DataFrame:
    """Genera una muestra 'actual' a partir de la referencia con ligero drift simulado."""
    rng = np.random.default_rng(random_state)
    current = reference_df.copy()
    num_cols = current.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        noise = rng.normal(0, current[col].std() * drift_factor, size=len(current))
        current[col] = current[col] + noise
    return current.sample(frac=0.8, random_state=random_state).reset_index(drop=True)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numerical_cols: List[str],
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.1,
) -> Tuple[Dict, List[str]]:
    """Detecta drift en las columnas numéricas indicadas."""
    report = {}
    drifted = []
    for col in numerical_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        psi = _psi(reference[col], current[col])
        ks = _ks(reference[col], current[col])
        status = "drift" if psi > psi_threshold or ks > ks_threshold else "ok"
        if status == "drift":
            drifted.append(col)
        report[col] = {
            "psi": round(psi, 4),
            "ks": round(ks, 4),
            "status": status,
        }
    return report, drifted


def run_drift_report(output_path: str = "monitoring/drift_report.json") -> Dict:
    """Ejecuta el análisis de drift completo y guarda el reporte."""
    clients_ref = pd.read_csv("data/clients.csv")
    tx_ref = pd.read_csv("data/transactions.csv")

    clients_cur = _synthetic_current(clients_ref)
    tx_cur = _synthetic_current(tx_ref)

    risk_report, risk_drifted = detect_drift(clients_ref, clients_cur, RISK_NUMERIC_COLS)
    fraud_report, fraud_drifted = detect_drift(tx_ref, tx_cur, FRAUD_NUMERIC_COLS)

    summary = {
        "drift_detected": bool(risk_drifted or fraud_drifted),
        "risk_drifted_features": risk_drifted,
        "fraud_drifted_features": fraud_drifted,
        "risk_report": risk_report,
        "fraud_report": fraud_report,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
