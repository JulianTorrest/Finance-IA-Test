# Presentación - Prueba Técnica Data Scientist Senior

Diapositivas en formato Markdown. Cada `##` representa una slide.

---

## 1. Portada

**Prueba Técnica Data Scientist Senior**

Solución de ciencia de datos productiva para riesgo financiero y fraude con agente IA.

- **Candidato:** Julian Torres
- **Fecha:** Octubre 2026
- **Repositorio:** https://github.com/JulianTorrest/Finance-IA-Test

---

## 2. Problema de negocio

Un banco del sector financiero necesita evolucionar su plataforma analítica hacia:

1. **Scoring de riesgo dinámico** de clientes.
2. **Detección de fraude y anomalías** en transacciones.
3. **Agente inteligente IA** que traduzca análisis a acciones concretas.

El objetivo no es solo predecir, sino **explicar, accionar y escalar** la solución.

---

## 3. Arquitectura de la solución

```
Datos dummy  →  Entrenamiento  →  Serving (FastAPI)  →  UI (Streamlit)  →  Agente IA
                 Model Registry        Predict API          Métricas        Mistral/Groq
                 Retraining            Drift PSI/KS
```

- Código modular: `src/`, `api/`, `app/`, `tests/`, `docs/`.
- Configuración centralizada en `config/config.yaml`.
- Secretos en `.streamlit/secrets.toml` y `.env`.

---

## 4. Datos y generación

- **5.000 clientes**, ~10.000 cuentas y **100.000 transacciones** generados sintéticamente.
- Variables demográficas, financieras y transaccionales.
- Targets: `risk_high` para riesgo y `fraud_real` para fraude.
- Reproducible con `src/generate_data.py`.

---

## 5. Modelos de machine learning

### Riesgo financiero
- Random Forest, Gradient Boosting, Logistic Regression.
- Variables: edad, ingreso, deuda, score crediticio, etc.
- Métrica: **ROC-AUC**.

### Fraude transaccional
- Isolation Forest (no supervisado) y Random Forest (supervisado).
- Variables: monto, hora, Z-score, velocidad, riesgo país.
- Métricas: **precision, recall, F1**.

---

## 6. Catálogo de modelos seleccionables

- Todos los modelos se registran en `models/model_catalog.json`.
- El usuario elige el modelo en Streamlit.
- FastAPI permite seleccionar modelo por query param.
- El LLM compara predicciones de múltiples modelos y explica discrepancias.

---

## 7. Agente IA

- Integración con **Mistral** y **Groq** con fallback automático.
- Modos: Experto, Ejecutivo, Técnico, Alerta operativa.
- Contexto enriquecido: perfil, cuentas, transacciones recientes, predicciones.
- Respuesta en markdown legible con secciones estructuradas.

---

## 8. Ciclo de vida del modelo (MLOps)

- **Entrenamiento:** `src/train_models.py` y `src/model_catalog.py`.
- **Registro:** `models/model_registry.json` y `models/model_catalog.json`.
- **Retraining:** `src/retrain.py` con detección por antigüedad y botón en UI.
- **Drift:** `src/drift.py` con PSI y KS, reporte y tab en Streamlit.
- **Monitoreo:** tab de métricas con tarjetas y gráficos.

---

## 9. Tests automatizados

- 14 tests con `pytest`.
- Cobertura:
  - Existencia y esquema de datos.
  - Carga de modelos.
  - Predicciones y contexto enriquecido.
- Ejecución: `pytest -q`.

---

## 10. FastAPI

Endpoints disponibles:

- `GET /health`
- `GET /catalog`
- `GET /predict/risk/{client_id}?model=...`
- `GET /predict/fraud/{transaction_id}?model=...`
- `GET /context/client/{client_id}`
- `GET /context/transaction/{transaction_id}`
- `POST /retrain?force=true`
- `GET /drift`

Permite consumo por otros sistemas del banco.

---

## 11. Métricas y valor de negocio

| KPI | Valor esperado |
|-----|----------------|
| ROC-AUC riesgo | > 0.95 |
| F1 fraude | Mejorable con tuning |
| Tiempo respuesta LLM | < 2s |
| Cobertura scoring | 100% clientes |
| Transacciones analizadas | 100.000 |

---

## 12. Supuestos y decisiones técnicas

- Datos sintéticos; en producción vendrían del data warehouse.
- Target de riesgo construido con reglas de negocio.
- Tasa de fraude ~2% para la demo.
- Mistral/Groq requieren API keys de tier adecuado.
- Se priorizó explicabilidad y facilidad de mantenimiento.

---

## 13. Demo / mockups

La aplicación Streamlit incluye:

- Tab **Riesgo Financiero** con selección de modelo.
- Tab **Fraude** con scatter de anomalías.
- Tab **Agente IA** con respuestas markdown.
- Tab **Monitoreo** con métricas y retraining.
- Tab **Drift** con PSI y KS.

---

## 14. Roadmap de mejoras

A corto plazo:
- A/B testing y shadow mode.
- SHAP para explicabilidad.
- Dockerización.

A mediano plazo:
- Feature store.
- MLflow para trazabilidad.
- CI/CD con GitHub Actions.

A largo plazo:
- Kubernetes / cloud.
- Multitenencia.
- Modelos más avanzados (XGBoost, autoencoders).

---

## 15. Cierre

La solución entrega:

- ✅ Datos, modelos y UI funcionales.
- ✅ Agente IA con fallback y comparación de modelos.
- ✅ MLOps: registro, retraining, drift.
- ✅ API REST para integración.
- ✅ Tests automatizados y documentación.

**Gracias.**
