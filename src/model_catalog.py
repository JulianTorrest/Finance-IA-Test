"""
Catálogo de modelos entrenados para riesgo y fraude.
Entrena alternativas y permite al usuario / API elegir cuál usar.
"""
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

with open("config/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def _risk_features():
    return CFG["models"]["risk"]["features"]


def _fraud_features():
    return CFG["models"]["fraud"]["features"]


def _preprocessor(features):
    return ColumnTransformer([("scaler", StandardScaler(), features)], remainder="passthrough")


def _save_and_log(name, artifact, metrics, kind):
    return {
        "name": name,
        "artifact": artifact,
        "metrics": metrics,
        "type": kind,
        "trained_at": datetime.now().isoformat(),
    }


def train_risk_random_forest(X_train, X_test, y_train, y_test):
    features = _risk_features()
    model = Pipeline([
        ("preprocess", _preprocessor(features)),
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
    artifact = "models/risk_random_forest_v1.0.0.joblib"
    joblib.dump(model, artifact)
    return _save_and_log("Riesgo - Random Forest", artifact, metrics, "risk")


def train_risk_gradient_boosting(X_train, X_test, y_train, y_test):
    features = _risk_features()
    model = Pipeline([
        ("preprocess", _preprocessor(features)),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
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
    artifact = "models/risk_gradient_boosting_v1.0.0.joblib"
    joblib.dump(model, artifact)
    return _save_and_log("Riesgo - Gradient Boosting", artifact, metrics, "risk")


def train_risk_logistic_regression(X_train, X_test, y_train, y_test):
    features = _risk_features()
    model = Pipeline([
        ("preprocess", _preprocessor(features)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
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
    artifact = "models/risk_logistic_regression_v1.0.0.joblib"
    joblib.dump(model, artifact)
    return _save_and_log("Riesgo - Logistic Regression", artifact, metrics, "risk")


def train_fraud_isolation_forest(X):
    features = _fraud_features()
    model = IsolationForest(
        n_estimators=100,
        contamination=CFG["models"]["fraud"]["contamination"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    preds = (model.predict(X) == -1).astype(int)
    df = pd.read_csv("data/transactions.csv")
    metrics = {
        "precision": round(precision_score(df["fraud_real"], preds, zero_division=0), 4),
        "recall": round(recall_score(df["fraud_real"], preds, zero_division=0), 4),
        "f1": round(f1_score(df["fraud_real"], preds, zero_division=0), 4),
    }
    artifact = "models/fraud_isolation_forest_v1.0.0.joblib"
    joblib.dump(model, artifact)
    return _save_and_log("Fraude - Isolation Forest", artifact, metrics, "fraud")


def train_fraud_random_forest(X_train, X_test, y_train, y_test):
    features = _fraud_features()
    model = Pipeline([
        ("preprocess", _preprocessor(features)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
    }
    artifact = "models/fraud_random_forest_v1.0.0.joblib"
    joblib.dump(model, artifact)
    return _save_and_log("Fraude - Random Forest", artifact, metrics, "fraud")


def train_all():
    os.makedirs("models", exist_ok=True)

    # Riesgo
    clients = pd.read_csv("data/clients.csv")
    X = clients[_risk_features()]
    y = clients[CFG["models"]["risk"]["target"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    risk_models = [
        train_risk_random_forest(X_train, X_test, y_train, y_test),
        train_risk_gradient_boosting(X_train, X_test, y_train, y_test),
        train_risk_logistic_regression(X_train, X_test, y_train, y_test),
    ]

    # Fraude
    tx = pd.read_csv("data/transactions.csv")
    X_fraud = tx[_fraud_features()].fillna(0)
    y_fraud = tx["fraud_real"]
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

    fraud_models = [
        train_fraud_isolation_forest(X_fraud),
        train_fraud_random_forest(Xf_train, Xf_test, yf_train, yf_test),
    ]

    catalog = {
        "project": CFG["project"]["name"],
        "version": CFG["project"]["version"],
        "trained_at": datetime.now().isoformat(),
        "risk_default": "Riesgo - Random Forest",
        "fraud_default": "Fraude - Isolation Forest",
        "risk_models": {m["name"]: m for m in risk_models},
        "fraud_models": {m["name"]: m for m in fraud_models},
    }

    with open("models/model_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(json.dumps(catalog, indent=2))
    return catalog


if __name__ == "__main__":
    train_all()
