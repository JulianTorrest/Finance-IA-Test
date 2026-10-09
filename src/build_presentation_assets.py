"""Genera assets visuales para la presentacion PDF."""
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = "docs/imgs"


def arquitectura():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (1, 2.2, "Datos\ndummy"),
        (3, 2.2, "Feature\nEngineering"),
        (5, 2.2, "Entrenamiento\nModelos"),
        (7, 2.2, "Model\nRegistry"),
        (3, 0.8, "FastAPI\nREST"),
        (7, 0.8, "Streamlit\nUI"),
        (5, 0.8, "Agente\nIA"),
    ]

    for x, y, label in boxes:
        rect = plt.Rectangle((x - 0.4, y - 0.35), 0.8, 0.7, color="#003366", ec="black", lw=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    arrows = [
        (1.4, 2.2, 2.6, 2.2),
        (3.4, 2.2, 4.6, 2.2),
        (5.4, 2.2, 6.6, 2.2),
        (5, 1.85, 5, 1.15),
        (5.4, 0.8, 6.6, 0.8),
        (3.4, 0.8, 4.6, 0.8),
        (3.8, 0.8, 4.6, 0.8),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#555555", lw=2))

    ax.set_title("Arquitectura de la solucion", fontsize=16, fontweight="bold", color="#003366", pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/arquitectura.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def metricas_riesgo():
    modelos = ["Random\nForest", "Gradient\nBoosting", "Logistic\nRegression"]
    valores = [0.9737, 0.9606, 0.9761]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#003366", "#00509e", "#4a90e2"]
    bars = ax.bar(modelos, valores, color=colors, edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Metricas de riesgo - ROC-AUC", fontsize=14, fontweight="bold", color="#003366")
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/metricas_riesgo.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def metricas_fraude():
    modelos = ["Isolation\nForest", "Random\nForest"]
    valores = [0.2491, 0.092]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#003366", "#00509e"]
    bars = ax.bar(modelos, valores, color=colors, edgecolor="black")
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Metricas de fraude - F1", fontsize=14, fontweight="bold", color="#003366")
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/metricas_fraude.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def cobertura():
    labels = ["Clientes", "Cuentas", "Transacciones"]
    valores = [5000, 9997, 100000]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#003366", "#00509e", "#4a90e2"]
    bars = ax.bar(labels, valores, color=colors, edgecolor="black")
    ax.set_ylabel("Cantidad", fontsize=12)
    ax.set_title("Cobertura de datos dummy", fontsize=14, fontweight="bold", color="#003366")
    ax.set_yscale("log")
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.2, f"{val:,}", ha="center", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/cobertura.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def cronograma():
    fases = ["Analisis", "Datos", "Modelado", "UI + LLM", "MLOps + API", "Sustentacion"]
    inicios = [0, 2, 3, 5, 7, 9]
    duraciones = [2, 1, 2, 2, 2, 1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#003366", "#00509e", "#4a90e2", "#5fa8f3", "#89c2f5", "#b3d9f7"]
    for i, (fase, ini, dur, color) in enumerate(zip(fases, inicios, duraciones, colors)):
        ax.barh(i, dur, left=ini, color=color, edgecolor="black", height=0.6)
        ax.text(ini + dur / 2, i, fase, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(fases)))
    ax.set_yticklabels([])
    ax.set_xlabel("Dias", fontsize=12)
    ax.set_title("Cronograma estimado (dias)", fontsize=14, fontweight="bold", color="#003366")
    ax.set_xlim(0, 11)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/cronograma.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def mlops_checklist():
    items = ["Registro", "Versionamiento", "Tests", "Drift", "Retraining", "API", "Docs"]
    valores = [1, 1, 1, 1, 1, 1, 1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#003366"] * len(items)
    bars = ax.barh(items, valores, color=colors, edgecolor="black", height=0.5)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([])
    ax.set_title("Checklist MLOps implementado", fontsize=14, fontweight="bold", color="#003366")
    for bar, item in zip(bars, items):
        ax.text(1.05, bar.get_y() + bar.get_height() / 2, "(OK)", va="center", fontsize=12, fontweight="bold", color="#003366")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/mlops_checklist.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def build_all():
    arquitectura()
    metricas_riesgo()
    metricas_fraude()
    cobertura()
    cronograma()
    mlops_checklist()
    print("Assets visuales generados en docs/imgs/")


if __name__ == "__main__":
    build_all()
