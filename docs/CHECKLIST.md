# Checklist - Prueba Técnica Data Scientist Senior

Basado en el documento `Prueba Técnica CD Sr.pdf`. Estado: **hecho**, **parcial** o **pendiente**.

## 1. Entendimiento del problema y necesidad

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Contexto bancario claro | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 1 |
| Dos modelos: riesgo y fraude | ✅ Hecho | `src/train_models.py`, `src/predict.py` |
| Orientación a agente IA | ✅ Hecho | `app/streamlit_app.py`, `src/llm_agent.py` |

## 2. Solidez del diseño (funcionamiento, costo, escalabilidad, performance)

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Arquitectura explicada | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 2 |
| Código modular (generación, train, predict, agente, app) | ✅ Hecho | `src/` y `app/` |
| Escalabilidad y costo mencionados | ✅ Hecho | `docs/ARQUITECTURA.md` |
| Performance aceptable para demo | ✅ Hecho | `models/*.joblib` serializados |

## 3. Calidad del planteamiento del código

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Código organizado | ✅ Hecho | Estructura de carpetas |
| Configuración centralizada | ✅ Hecho | `config/config.yaml` |
| API de inferencia reutilizable | ✅ Hecho | `src/predict.py` |
| Buenas prácticas básicas | ✅ Hecho | `.gitignore`, `.env.example`, `requirements.txt` |
| Tests unitarios | ⚠️ Parcial | No hay suite de tests automatizados |

## 4. Preparación y metodología de entornos

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Entorno virtual creado | ✅ Hecho | `.venv/` |
| Requirements definido | ✅ Hecho | `requirements.txt` |
| Datos dummy | ✅ Hecho | `src/generate_data.py`, `data/*.csv` |

## 5. Ciclo de vida del modelo (MLOps)

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Entrenamiento y serialización | ✅ Hecho | `src/train_models.py` |
| Registro de modelos | ✅ Hecho | `models/model_registry.json` |
| Versionamiento | ✅ Hecho | Nombres `*_v1.0.0.joblib` y registry |
| Monitoreo de métricas | ✅ Hecho | Tab Monitoreo en Streamlit |
| Estrategia de retraining | ⚠️ Parcial | Documentada en `ARQUITECTURA.md`, no automatizada |
| Data drift / concept drift | ⚠️ Parcial | Mencionado, no implementado con librería |

## 6. Tiempos estimados, entregables y mejoras en el ciclo de vida

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Estimación de tiempos | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 9 |
| Entregables definidos | ✅ Hecho | README + docs |
| Mejoras en ciclo de vida | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 10 |

## 7. Métricas clave y valor de negocio

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Métricas de riesgo (ROC-AUC) | ✅ Hecho | `models/model_registry.json`, tab Monitoreo |
| Métricas de fraude (F1) | ✅ Hecho | `models/model_registry.json`, tab Monitoreo |
| Valor de negocio explicado | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 5 |

## 8. Capacidad de comunicar decisiones técnicas

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Documentación de decisiones | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 4, 7, 8 |
| README ejecutable | ✅ Hecho | `README.md` |

## 9. Supuestos específicos coherentes con la realidad

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Supuestos documentados | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 6 |

## 10. Mockups de salidas técnicas

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Dashboard de riesgo | ✅ Hecho | Tab Riesgo Financiero |
| Dashboard de fraude | ✅ Hecho | Tab Fraude |
| Agente IA interactivo | ✅ Hecho | Tab Agente IA |
| Panel de monitoreo | ✅ Hecho | Tab Monitoreo |

## 11. Posibles mejoras o features futuras

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Roadmap de mejoras | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 10 |

## 12. Preparación para sustentación

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| Qué priorizar si presionan a producción | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 11 |
| Alternativas descartadas | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 7 |
| Riesgos aceptados | ✅ Hecho | `docs/ARQUITECTURA.md` secc. 8 |

## Resumen

- **Totalmente hecho:** ~85%
- **Parcial o mejorable:** tests automatizados, retraining automático, drift monitoring avanzado, API REST formal.
- **Pendiente para refinar:** ajustar prompts del agente según tier de LLM, pulir UI, agregar tests.
