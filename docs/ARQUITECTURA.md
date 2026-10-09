# Documento de Arquitectura y Decisiones Técnicas

## 1. Entendimiento del problema

El banco necesita modernizar su plataforma analítica con dos capacidades críticas:

1. **Riesgo financiero cliente (scoring dinámico):** evaluar continuamente la probabilidad de que un cliente genere pérdida financiera, permitiendo acciones preventivas (revisión de cupos, contacto, reestructuración, etc.).
2. **Detección de fraude/anomalías transaccional:** identificar en tiempo o batch operaciones atípicas que puedan ser fraude, abuso o riesgo operacional.

Adicionalmente, la organización busca una **capa de agente IA** que traduzca los resultados analíticos a lenguaje natural y acciones concretas para negocio.

## 2. Arquitectura propuesta

```
┌─────────────────────────────────────────────────────────────┐
│                    FUENTES DE DATOS                         │
│  Core bancario · Data Lake · Transacciones · Customer 360   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│               Ingesta y preparación de datos                │
│   src/generate_data.py  →  data/*.csv                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│             Entrenamiento / Retraining (batch)              │
│   src/train_models.py  →  models/*.joblib                   │
│   model_registry.json  →  versionamiento y métricas         │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                API de inferencia / Serving                  │
│   src/predict.py  →  riesgo/fraude                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                 Aplicación y Agente IA                      │
│   app/streamlit_app.py  +  src/llm_agent.py                 │
│   Mistral / Groq  →  respuestas accionables                 │
└─────────────────────────────────────────────────────────────┘
```

### Componentes principales

| Componente | Responsabilidad | Tecnología |
|------------|-----------------|------------|
| `src/generate_data.py` | Generación de datos dummy reproducible | Pandas, NumPy |
| `src/train_models.py` | Entrenar y versionar modelos | Scikit-learn, Joblib |
| `src/predict.py` | API de inferencia reutilizable | Pandas, Joblib |
| `src/llm_agent.py` | Integración con LLMs | `mistralai`, `groq` |
| `app/streamlit_app.py` | UI/UX y consumo por usuarios | Streamlit, Plotly |
| `models/model_registry.json` | Registro de versiones y métricas | JSON manual |
| `config/config.yaml` | Configuración centralizada | YAML |

## 3. Modelos implementados

### 3.1 Riesgo financiero

- **Tipo:** clasificación supervisada (`RandomForestClassifier`).
- **Target:** `risk_high` generado a partir del score crediticio, endeudamiento y antigüedad.
- **Variables:** edad, ingreso, deuda, score, antigüedad, cantidad de productos, ratio deuda/ingreso.
- **Salida:** probabilidad de riesgo y explicación por importancia de variables.
- **Ciclo de vida:** reentrenamiento mensual/trimestral a medida que evolucionan los patrones de endeudamiento.

### 3.2 Fraude transaccional

- **Tipo:** detección de anomalías (`IsolationForest`) + anotación sintética de fraude.
- **Variables:** monto, hora, día de la semana, canal, riesgo país, velocidad horaria, z-score del monto.
- **Salida:** score de anomalía y bandera binaria.
- **Ciclo de vida:** reentrenamiento frecuente ante nuevos vectores de fraude; ajuste de `contamination` por estimación de tasa real.

## 4. Ciclo de vida del modelo (MLOps ligero)

1. **Desarrollo:** exploración, feature engineering y validación en datos dummy/históricos.
2. **Entrenamiento:** `src/train_models.py` genera artefactos y actualiza `models/model_registry.json`.
3. **Evaluación:** métricas ROC-AUC, precision, recall y F1.
4. **Promoción:** el registro guarda `status` (`development`, `staging`, `production`).
5. **Monitoreo:** pestaña de Streamlit y comparación periódica de métricas vs baseline.
6. **Retraining:** cuando PSI (Population Stability Index) de variables clave supere 0.2 o caída de F1 > 5%.

## 5. Métricas clave y valor de negocio

| KPI | Modelo | Valor esperado |
|-----|--------|----------------|
| ROC-AUC | Riesgo | > 0.80 (reduce impagos) |
| F1 fraude | Fraude | Mejora detección con menos falsos positivos |
| Tiempo de respuesta Agente IA | LLM | < 2 s por consulta |
| Cobertura de clientes scorificados | Riesgo | 100% de clientes activos |
| Tasa de transacciones revisadas | Fraude | Ajustable por umbral de negocio |

## 6. Supuestos

- Los datos son sintéticos para la prueba; en producción vendrían del data warehouse.
- El target de riesgo se construye con reglas de negocio; en la realidad usaría default/mora real.
- La tasa de fraude real se supone ~2% para la demostración.
- Se dispone de API keys para Mistral y Groq en variables de entorno.
- El sistema corre on-premise o en contenedores; la solución es portable.

## 7. Alternativas consideradas y descartadas

| Alternativa | ¿Por qué se descartó? |
|-------------|------------------------|
| XGBoost / LightGBM | Mayor poder predictivo, pero aumenta complejidad y dependencias; elegí RandomForest por simplicidad y explicabilidad inmediata. |
| Deep Learning para fraude | Requiere más datos y recursos; IsolationForest es un baseline robusto y rápido para anomalías. |
| FastAPI en vez de Streamlit | Streamlit agiliza la demo visual; FastAPI se propone como evolución para consumo API por otros sistemas. |
| MLflow / DVC | Son la dirección ideal para MLOps, pero para la prueba técnica se optó por un registro JSON manual para reducir complejidad de setup. |

## 8. Riesgos aceptados

- **Dependencia de APIs externas:** Mistral y Groq pueden tener latencia o caídas; se recomienda caché de respuestas y fallback a reglas.
- **Datos históricos limitados:** el scoring aprende de transversalidad, no de series de tiempo largas; se propone evolucionar a modelos de survival/cox.
- **Interpretación del LLM:** respuestas no deterministas; se debe auditar y validar con negocio.

## 9. Entregables y tiempo estimado

| Fase | Tiempo estimado | Entregable |
|------|-----------------|------------|
| Análisis y diseño | 1-2 días | Documento de arquitectura |
| Generación de datos y features | 1 día | Datos dummy y pipeline ETL |
| Modelado y validación | 2-3 días | Modelos entrenados y métricas |
| Integración LLM y UI | 2 días | Streamlit funcional |
| Documentación y sustentación | 1 día | README, docs, slides |

## 10. Mejoras futuras

- Servir modelos vía REST (FastAPI) para consumo masivo.
- Pipeline de CI/CD con GitHub Actions y pruebas unitarias.
- Feature store para reutilización de variables entre modelos.
- Explainability con SHAP en Streamlit.
- Data drift / concept drift con Evidently o custom monitoring.
- Historial de riesgo mensual para análisis temporal.

## 11. Decisiones si se presiona a producción

Si la dirección presiona a producción antes de tiempo, priorizaría:
1. **Modelo de fraude:** impacto directo en pérdidas y regulación.
2. **API de riesgo batch:** scorificación diaria de clientes nuevos.
3. **Guardrails del LLM:** prompts fijos, validación de salida y trazabilidad.
4. **Logging y monitoreo básico:** alertas de caída y desviación de predicciones.

Lo que se postergaría: retraining automático, A/B testing y feature store.

## 12. Mockup de salidas técnicas

El mockup se puede observar en la aplicación Streamlit:
- Tabla de clientes con riesgo ordenado.
- Gráfico de distribución de riesgo.
- Scatter monto vs anomalía.
- Panel de registro de modelos.
- Chat con agente IA.

En producción se complementaría con dashboards en Looker/PowerBI y alertas en Slack/Splunk.
