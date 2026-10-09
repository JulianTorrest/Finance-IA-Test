"""
Módulo de retraining automático. Determina si los modelos deben
reentrenarse por antigüedad o por detección de drift, y ejecuta
el pipeline de entrenamiento.
"""
import datetime
import json
import os

from src.train_models import update_registry


def should_retrain(days: int = 30, drift_detected: bool = False) -> bool:
    """Decide si se debe reentrenar por antigüedad o drift."""
    try:
        with open("models/model_registry.json", "r", encoding="utf-8") as f:
            registry = json.load(f)
        last = datetime.datetime.fromisoformat(registry["trained_at"])
        age_days = (datetime.datetime.now() - last).days
        return age_days > days or drift_detected
    except Exception:
        return True


def retrain(force: bool = False, drift_detected: bool = False) -> str:
    """Ejecuta el retraining si las condiciones lo indican."""
    if force or should_retrain(drift_detected=drift_detected):
        update_registry()
        return "Reentrenamiento completado"
    return "No es necesario reentrenar"


def auto_retrain_if_needed():
    """Punto de entrada para cron o scheduler."""
    if should_retrain():
        return retrain()
    return "Modelos vigentes"


if __name__ == "__main__":
    print(auto_retrain_if_needed())
