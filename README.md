# Prueba Técnica Data Scientist Senior - Banca Analítica

Solución productiva de ciencia de datos para un banco que busca evolucionar su plataforma analítica con modelos predictivos y un agente inteligente IA.

## Alcance

- **Modelo de riesgo financiero:** scoring de riesgo de clientes actualizable en el tiempo.
- **Modelo de fraude:** detección de transacciones atípicas con anomalías.
- **Agente IA:** consumo de resultados vía LLMs Mistral y Groq para respuestas accionables.
- **Ciclo de vida del modelo:** registro, versionamiento, monitoreo y control.
- **Datos dummy:** clientes, cuentas y transacciones generados sintéticamente.

## Estructura del proyecto

```
prueba_tecnica_cd_sr/
├── app/                  # Aplicación Streamlit
├── config/               # Configuración centralizada
├── data/                 # Datos dummy
├── docs/                 # Documentación arquitectónica
├── models/               # Artefactos y registro de modelos
├── src/                  # Código fuente
│   ├── generate_data.py  # Generación de datos dummy
│   ├── train_models.py   # Entrenamiento y registro
│   ├── predict.py        # API de inferencia
│   └── llm_agent.py      # Integración Mistral/Groq
├── tests/                # Pruebas
├── .env.example
├── config.yaml
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

Copiar `.env.example` a `.env` y agregar las API keys:

```bash
cp .env.example .env
```

## Uso

### 1. Generar datos dummy

```bash
python src/generate_data.py
```

### 2. Entrenar modelos

```bash
python src/train_models.py
```

### 3. Ejecutar la aplicación Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Tecnologías

- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas / NumPy
- Groq y Mistral API
- Joblib para serialización
- YAML para configuración

## Contacto / Autor

Solución preparada para sustentación técnica.
