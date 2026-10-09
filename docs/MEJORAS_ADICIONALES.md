# Mejoras adicionales propuestas (fuera del alcance de la prueba técnica)

Estas mejoras no estaban explícitamente en el documento `Prueba Técnica CD Sr.pdf`, pero aportarían valor real en una puesta en producción y elevan el nivel de madurez de la solución.

## 1. CI/CD y automatización

- GitHub Actions para ejecutar tests, lint y `pytest` en cada push.
- Pipeline de entrenamiento automático cuando se detecte drift o se suban nuevos datos.
- Versionado de artefactos con DVC o MLflow.

## 2. Contenerización y orquestación

- `Dockerfile` para Streamlit y otro para FastAPI.
- `docker-compose.yml` para levantar datos, API y UI localmente.
- Kubernetes (EKS/GKE/AKS) para escalar el serving y el retraining.

## 3. Feature store

- Centralizar variables de riesgo y fraude en un feature store (Feast, Tecton, SageMaker Feature Store).
- Reutilización de features entre modelos y consistencia entre entrenamiento e inferencia.

## 4. Explainability y fairness

- SHAP o LIME para explicar predicciones individuales a negocio y auditoría.
- Análisis de sesgo por segmento, género o región.
- Reportes de fairness (demographic parity, equalized odds).

## 5. Observabilidad y logging

- MLflow, Weights & Biases o Neptune para trazabilidad de experimentos.
- Prometheus + Grafana para monitoreo de latencia, throughput y errores.
- Logs estructurados en JSON para Splunk, ELK o Datadog.

## 6. A/B testing y shadow mode

- Comparar el nuevo modelo contra el actual en paralelo (shadow mode).
- Estrategias de asignación aleatoria para evaluar impacto real de negocio.

## 7. Seguridad y gobierno

- Autenticación y autorización en FastAPI (OAuth2, JWT, API keys por cliente).
- Cifrado de datos sensibles y tokenización de IDs.
- Control de acceso basado en roles para el tab de monitoreo.

## 8. Caché y optimización de LLM

- Caché de respuestas frecuentes del agente IA para reducir costos.
- Selección dinámica del modelo de lenguaje según complejidad de la consulta.
- Streaming de respuestas en tiempo real.

## 9. Data versioning y calidad

- DVC para versionar datasets y modelos.
- Great Expectations o Pandera para validar esquema y calidad de datos.

## 10. Notificaciones y alertas

- Alertas por Slack, correo o SMS cuando se detecte drift o fraude alto.
- Dashboards operativos en PowerBI, Looker o Tableau.

## 11. Multitenencia y personalización

- Soporte para múltiples bancos o líneas de negocio con configuraciones propias.
- Umbrales de riesgo y fraude personalizables por segmento.

## 12. Modelos más avanzados

- XGBoost / LightGBM para riesgo con búsqueda de hiperparámetros.
- Autoencoders o GANs para fraude cuando haya más datos etiquetados.
- Modelos de survival o series de tiempo para riesgo dinámico.
