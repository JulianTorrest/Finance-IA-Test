# Prueba Técnica Data Scientist Senior

---

## 1. Presentación

**Solución de ciencia de datos productiva para riesgo financiero y fraude con agente IA**

- **Proyecto:** Finance IA Test
- **Repositorio:** https://github.com/JulianTorrest/Finance-IA-Test
- **Fecha:** Octubre 2026

---

## 2. Objetivos del proyecto

1. Construir un modelo de **riesgo financiero** que evolucione con el cliente.
2. Detectar **comportamientos transaccionales atípicos** con enfoque de fraude.
3. Habilitar un **agente inteligente IA** que traduzca análisis en acciones.
4. Aplicar buenas prácticas de **ciclo de vida de modelos (MLOps)**.
5. Exponer resultados vía **Streamlit y FastAPI** para usuarios y sistemas.

---

## 3. Cronograma

| Fase | Duración | Entregable |
|------|----------|------------|
| Análisis y diseño | 1-2 días | Arquitectura y supuestos |
| Datos y features | 1 día | Datos dummy y pipeline ETL |
| Modelado | 2-3 días | Modelos entrenados y catálogo |
| Integración LLM y UI | 2 días | Streamlit y agente IA |
| MLOps y API | 2 días | Tests, drift, retraining, FastAPI |
| Sustentación | 1 día | Presentación y documentación |

**Total estimado:** 9-11 días hábiles.

---

## 4. Arquitectura

```
Datos dummy  →  Feature engineering  →  Entrenamiento  →  Model Registry
                                   ↓                    ↓
                            Streamlit UI         FastAPI REST
                                   ↓
                          Agente IA (Mistral + Groq)
```

Capas principales:
- **Ingesta y preparación:** `src/generate_data.py`
- **Entrenamiento:** `src/model_catalog.py`, `src/train_models.py`
- **Serving:** `src/predict.py` y `api/main.py`
- **Experiencia:** `app/streamlit_app.py`
- **MLOps:** `src/retrain.py`, `src/drift.py`, `tests/`

---

## 5. Modelos desarrollados

### Riesgo financiero
- Random Forest
- Gradient Boosting
- Logistic Regression

### Fraude transaccional
- Isolation Forest (no supervisado)
- Random Forest (supervisado)

**El usuario puede seleccionar el modelo en la UI y en la API.**

---

## 6. Resultados

### Métricas de modelos

| Modelo | Métrica principal | Valor |
|--------|-------------------|-------|
| Riesgo - Random Forest | ROC-AUC | 0.974 |
| Riesgo - Logistic Regression | ROC-AUC | 0.976 |
| Fraude - Isolation Forest | F1 | 0.249 |
| Fraude - Random Forest | F1 | 0.092 |

### Cobertura
- 5.000 clientes analizados.
- 100.000 transacciones evaluadas.
- 100% de clientes scorificados.

---

## 7. Impacto de negocio

- **Reducción de pérdidas:** detección temprana de clientes de alto riesgo.
- **Eficiencia operativa:** fraude detectado automáticamente para revisión.
- **Toma de decisiones:** el agente IA entrega recomendaciones accionables en lenguaje natural.
- **Escalabilidad:** FastAPI permite integración con canales digitales del banco.
- **Governanza:** registro, versionamiento, drift y retraining incluidos.

---

## 8. Agente IA

- Integración con **Mistral** y **Groq** con fallback.
- Responde en markdown con secciones: resumen, factores, acción, urgencia, riesgo, justificación.
- **Compara predicciones** de múltiples modelos de ML.
- Contexto enriquecido: cliente, cuentas, transacciones recientes, scores.

---

## 9. MLOps implementado

| Capacidad | Estado |
|-----------|--------|
| Registro de modelos | (OK) |
| Versionamiento | (OK) |
| Tests automatizados (pytest) | (OK) |
| Drift monitoring (PSI/KS) | (OK) |
| Retraining automático | (OK) |
| API REST (FastAPI) | (OK) |
| Documentación | (OK) |

---

## 10. Supuestos clave

- Datos sintéticos para la demo; en producción vendrían del data warehouse.
- Target de riesgo construido con reglas de negocio.
- Tasa de fraude simulada ~2%.
- API keys de Mistral/Groq en tier gratuito.

---

## 11. Roadmap

**Corto plazo**
- SHAP para explicabilidad.
- A/B testing y shadow mode.
- Dockerización.

**Mediano plazo**
- Feature store.
- MLflow / experiment tracking.
- CI/CD con GitHub Actions.

**Largo plazo**
- Kubernetes / cloud deployment.
- Multitenencia.
- XGBoost y autoencoders.

---

## 12. Cierre

La solución entrega:

- (OK) Datos, modelos y UI funcionales.
- (OK) Catálogo de modelos seleccionables.
- (OK) Agente IA con fallback y comparación de modelos.
- (OK) MLOps: registro, drift, retraining y tests.
- (OK) API REST para integración con otros sistemas.
- (OK) Presentación ejecutiva lista para sustentación.

**¡Gracias!**
