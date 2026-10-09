"""
Agente IA avanzado: orquestación de Mistral y Groq con fallback,
prompts enriquecidos y salida estructurada.
"""
import json
import os
from typing import Dict, List


def _get_key(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name) or st.secrets["api_keys"].get(name)
    except Exception:
        return None


def _build_system_prompt(mode: str = "expert") -> str:
    base = (
        "Eres un director de riesgo y fraude de un banco con 20 años de experiencia. "
        "Analizas datos de clientes y transacciones para emitir juicios claros, accionables y justificados. "
        "Responde siempre en español, con tono profesional y directo.\n\n"
        "INSTRUCCIONES:\n"
        "1. Usa el contexto entregado (perfil del cliente, riesgo, transacciones, anomalías).\n"
        "2. Si el contexto incluye predicciones de varios modelos (predicciones_riesgo o predicciones_fraude), "
        "compara brevemente los resultados e indica si los modelos concuerdan o discrepan y por qué.\n"
        "3. Razona paso a paso antes de concluir.\n"
        "4. Devuelve SIEMPRE la respuesta en formato markdown con las secciones indicadas abajo. "
        "NO uses bloques de código JSON, ni triple backticks, ni escapes. "
        "El texto debe ser directamente legible para un usuario funcional del banco.\n\n"
        "FORMATO OBLIGATORIO:\n\n"
        "# Resumen ejecutivo\n"
        "2-3 oraciones con la conclusión principal.\n\n"
        "# Factores clave\n"
        "- Factor 1\n"
        "- Factor 2\n"
        "- Factor 3\n\n"
        "# Acción recomendada\n"
        "Qué debe hacer el banco con este cliente o transacción.\n\n"
        "# Urgencia\n"
        "alta | media | baja\n\n"
        "# Riesgo score\n"
        "Un número entre 0.0 (ningún riesgo) y 1.0 (riesgo extremo).\n\n"
        "# Riesgo monetario estimado\n"
        "Descripción cualitativa o estimación del impacto.\n\n"
        "# Justificación\n"
        "Explicación técnica breve de por qué se toma la decisión.\n\n"
        "# Preguntas de seguimiento\n"
        "- Pregunta 1\n"
        "- Pregunta 2\n\n"
        "4. Si la información es insuficiente, usa la sección 'Preguntas de seguimiento' para solicitar datos.\n"
    )

    if mode == "executive":
        base += (
            "\nMODO EJECUTIVO: sé breve. Máximo 150 palabras en el resumen. "
            "Prioriza la acción recomendada y el impacto de negocio."
        )
    elif mode == "technical":
        base += (
            "\nMODO TÉCNICO: incluye análisis detallado de variables, scores y anomalías. "
            "Explica patrones estadísticos relevantes."
        )
    elif mode == "alert":
        base += (
            "\nMODO ALERTA: actúa como un operador de monitoreo. Indica si se debe alertar, "
            "investigar, bloquear o escalar la operación de inmediato."
        )
    else:
        base += (
            "\nMODO EXPERTO: balance entre detalle técnico y claridad ejecutiva. "
            "Sé exhaustivo pero directo."
        )

    return base


def _build_messages(
    context: Dict,
    question: str,
    mode: str,
    chat_history: List[Dict] = None,
) -> List[Dict]:
    system = _build_system_prompt(mode)
    user = (
        "=== CONTEXTO DEL CLIENTE / TRANSACCIÓN ===\n"
        f"{json.dumps(context, indent=2, ensure_ascii=False)}\n\n"
        f"=== PREGUNTA DEL USUARIO ===\n{question}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if chat_history:
        messages = [{"role": "system", "content": system}] + chat_history + [{"role": "user", "content": user}]
    return messages


def _call_groq(messages: List[Dict], model: str = "llama3-8b-8192") -> str:
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("La librería groq no está instalada.")

    api_key = _get_key("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY no configurada.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def _call_mistral(messages: List[Dict], model: str = "open-mistral-nemo") -> str:
    try:
        from mistralai import Mistral
    except ImportError:
        raise RuntimeError("La librería mistralai no está instalada.")

    api_key = _get_key("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY no configurada.")

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def ask(
    context: Dict,
    question: str,
    provider: str = "auto",
    mode: str = "expert",
    chat_history: List[Dict] = None,
    groq_model: str = "llama3-8b-8192",
    mistral_model: str = "open-mistral-nemo",
) -> str:
    messages = _build_messages(context, question, mode, chat_history)

    if provider == "groq":
        return _call_groq(messages, model=groq_model)
    if provider == "mistral":
        return _call_mistral(messages, model=mistral_model)

    errors = {}
    for attempt in ["groq", "mistral"]:
        try:
            if attempt == "groq":
                return _call_groq(messages, model=groq_model)
            return _call_mistral(messages, model=mistral_model)
        except Exception as e:
            errors[attempt] = str(e)

    return (
        "Error: ambos proveedores fallaron. "
        f"Detalles: {errors}. "
        "Verifica que las API keys y los modelos estén disponibles en tu tier."
    )


def ask_parsed(*args, **kwargs) -> Dict:
    """Devuelve la respuesta del LLM parseada como JSON si es posible."""
    raw = ask(*args, **kwargs)
    try:
        # A veces el modelo envuelve JSON en bloques de código markdown
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except Exception:
        return {
            "resumen_ejecutivo": raw,
            "error_parseo": "El modelo no devolvió JSON estructurado",
        }
