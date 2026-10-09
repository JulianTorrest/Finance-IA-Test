"""
Entrenamiento y registro de los modelos de riesgo y fraude.
Versión dummy/demo con fines ilustrativos para la prueba técnica.
"""
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

with open("config/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def train_risk():
    cfg = CFG["models"]["risk"]
    df = pd.read_csv("data/clients.csv")
    X = df[cfg["features"]]
    y = df[cfg["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    preprocessor = ColumnTransformer([("scaler", StandardScaler(), num_features)], remainder="passthrough")

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)),
    ])
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "precision_1": round(report["1"]["precision"], 4),
        "recall_1": round(report["1"]["recall"], 4),
        "f1_1": round(report["1"]["f1-score"], 4),
    }

    joblib.dump(model, cfg["artifact"])
    return metrics


def train_fraud():
    cfg = CFG["models"]["fraud"]
    df = pd.read_csv("data/transactions.csv")
    X = df[cfg["features"]].fillna(0)

    model = IsolationForest(
        n_estimators=100,
        contamination=cfg["contamination"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    df["anomaly_score"] = -model.decision_function(X)
    df["fraud_predicted"] = (model.predict(X) == -1).astype(int)

    metrics = {
        "precision": round(precision_score(df["fraud_real"], df["fraud_predicted"], zero_division=0), 4),
        "recall": round(recall_score(df["fraud_real"], df["fraud_predicted"], zero_division=0), 4),
        "f1": round(f1_score(df["fraud_real"], df["fraud_predicted"], zero_division=0), 4),
    }

    joblib.dump(model, cfg["artifact"])
    df.to_csv("data/transactions_scored.csv", index=False)
    return metrics


def update_registry():
    os.makedirs("models", exist_ok=True)
    risk_metrics = train_risk()
    fraud_metrics = train_fraud()

    registry = {
        "project": CFG["project"]["name"],
        "version": CFG["project"]["version"],
        "trained_at": datetime.now().isoformat(),
        "models": {
            "risk": {
                "name": CFG["models"]["risk"]["name"],
                "version": CFG["models"]["risk"]["version"],
                "artifact": CFG["models"]["risk"]["artifact"],
                "status": "production",
                "metrics": risk_metrics,
            },
            "fraud": {
                "name": CFG["models"]["fraud"]["name"],
                "version": CFG["models"]["fraud"]["version"],
                "artifact": CFG["models"]["fraud"]["artifact"],
                "status": "production",
                "metrics": fraud_metrics,
            },
        },
    }

    with open("models/model_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    update_registry()
