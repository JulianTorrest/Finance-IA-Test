"""
Generación de datos dummy para la prueba técnica.
Simula un ecosistema bancario con clientes, cuentas y transacciones.
"""
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml


with open("config/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

rng = np.random.default_rng(CFG["data"]["random_state"])
random.seed(CFG["data"]["random_state"])


def generate_clients(n=5000):
    ids = [f"CLI{str(i).zfill(8)}" for i in range(1, n + 1)]
    ages = rng.integers(18, 80, n)
    monthly_incomes = rng.lognormal(8.5, 0.8, n).astype(int)
    total_debts = rng.lognormal(9.0, 1.0, n).astype(int)
    credit_scores = rng.normal(650, 120, n).astype(int).clip(300, 900)
    tenures = rng.integers(0, 240, n)
    num_products = rng.integers(1, 6, n)
    types = rng.choice(["Persona", "Empresa"], n, p=[0.9, 0.1])
    countries = rng.choice(["CO", "US", "MX", "PE", "CL"], n, p=[0.85, 0.05, 0.05, 0.03, 0.02])
    segments = rng.choice(["Mass", "Premium", "Private"], n, p=[0.70, 0.25, 0.05])

    df = pd.DataFrame({
        "client_id": ids,
        "age": ages,
        "monthly_income": monthly_incomes,
        "total_debt": total_debts,
        "credit_score": credit_scores,
        "tenure_months": tenures,
        "num_products": num_products,
        "client_type": types,
        "country": countries,
        "segment": segments,
    })

    df["debt_to_income"] = (df["total_debt"] / (df["monthly_income"] * 12 + 1)).round(4)

    # Target de riesgo: alto si score bajo + deuda/ingreso alta + pocos productos
    risk_score = (
        (900 - df["credit_score"]) / 600 * 0.5
        + df["debt_to_income"].clip(0, 5) / 5 * 0.3
        + (1 - df["tenure_months"] / 240) * 0.1
        + (1 - df["num_products"] / 6) * 0.1
    )
    df["risk_high"] = (risk_score > rng.uniform(0.45, 0.75, n)).astype(int)

    return df


def generate_accounts(clients):
    rows = []
    for client_id in clients["client_id"]:
        n_accounts = rng.integers(1, 4)
        for _ in range(n_accounts):
            rows.append({
                "account_id": f"ACC{client_id[3:]}{rng.integers(100,999)}",
                "client_id": client_id,
                "balance": int(rng.lognormal(8.0, 1.2)),
                "account_type": rng.choice(["Ahorros", "Corriente", "CDT", "Nomina"], p=[0.50, 0.30, 0.12, 0.08]),
                "currency": "COP",
                "opened_months_ago": rng.integers(1, 120),
            })
    return pd.DataFrame(rows)


def generate_transactions(accounts, n=100_000):
    account_ids = accounts["account_id"].tolist()
    client_map = accounts.set_index("account_id")["client_id"].to_dict()

    start = datetime(2023, 1, 1)
    end = datetime(2024, 9, 30)
    delta = end - start

    tx_rows = []
    for i in range(n):
        acc = rng.choice(account_ids)
        ts = start + timedelta(seconds=int(rng.integers(0, int(delta.total_seconds()))))
        base_amount = rng.lognormal(11, 1.5)
        # 2% de transacciones con monto atípico (fraude simulado)
        is_fraud = rng.random() < 0.02
        if is_fraud:
            base_amount = base_amount * rng.uniform(5, 20)
        amount = int(base_amount)
        channel = rng.choice(["Web", "Mobile", "ATM", "Sucursal", "ACH"], p=[0.35, 0.35, 0.15, 0.05, 0.10])
        country = rng.choice(["CO", "US", "MX", "PE", "CL", "PA"], p=[0.75, 0.10, 0.06, 0.04, 0.03, 0.02])
        tx_rows.append({
            "transaction_id": f"TXN{str(i).zfill(10)}",
            "account_id": acc,
            "client_id": client_map[acc],
            "amount": amount,
            "type": rng.choice(["Débito", "Crédito", "Transferencia", "Pago"], p=[0.45, 0.25, 0.20, 0.10]),
            "channel": channel,
            "country": country,
            "timestamp": ts,
            "fraud_real": int(is_fraud),
        })

    df = pd.DataFrame(tx_rows)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Características derivadas: velocity_1h (transacciones por hora por cuenta)
    df = df.sort_values(["account_id", "timestamp"])
    df["velocity_1h"] = (
        df.groupby("account_id")
        .rolling("1h", on="timestamp")["transaction_id"]
        .count()
        .values
    )
    # amount zscore por cuenta
    df["amount_zscore"] = df.groupby("account_id")["amount"].transform(
        lambda x: ((x - x.mean()) / (x.std() + 1)).round(4)
    )
    # riesgo del país
    country_risk = {"CO": 0.2, "US": 0.1, "MX": 0.3, "PE": 0.4, "CL": 0.3, "PA": 0.6}
    df["country_risk_score"] = df["country"].map(country_risk)

    # canal codificado
    channel_map = {"Web": 0, "Mobile": 1, "ATM": 2, "Sucursal": 3, "ACH": 4}
    df["channel_encoded"] = df["channel"].map(channel_map)

    return df.reset_index(drop=True)


def main():
    out = CFG["data"]["output_dir"]
    os.makedirs(out, exist_ok=True)

    clients = generate_clients(CFG["data"]["n_clients"])
    accounts = generate_accounts(clients)
    transactions = generate_transactions(accounts, CFG["data"]["n_transactions"])

    clients.to_csv(f"{out}/clients.csv", index=False)
    accounts.to_csv(f"{out}/accounts.csv", index=False)
    transactions.to_csv(f"{out}/transactions.csv", index=False)

    print(f"Datos generados: {len(clients)} clientes, {len(accounts)} cuentas, {len(transactions)} transacciones")


if __name__ == "__main__":
    main()
