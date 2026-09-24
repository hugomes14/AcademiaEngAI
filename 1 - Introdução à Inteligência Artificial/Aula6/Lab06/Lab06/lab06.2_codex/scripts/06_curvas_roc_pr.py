# -*- coding: utf-8 -*-
"""
Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos
Etapa 06: Curvas ROC e Precision-Recall

Objetivo:
- Gerar curva ROC comparativa para todos os modelos.
- Gerar curva Precision-Recall comparativa, útil com classe positiva rara.
- Guardar gráficos em PNG e PDF.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

print("=" * 80)
print("ETAPA 06 — CURVAS ROC E PRECISION-RECALL")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / "artefactos" / "previsoes"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
GRAPH_DIR = BASE_DIR / "artefactos" / "graficos"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

training_log_path = MODEL_DIR / "registo_treino_modelos.csv"
if not training_log_path.exists():
    raise FileNotFoundError(f"Registo de treino não encontrado: {training_log_path}")

training_df = pd.read_csv(training_log_path)

# -------------------------------------------------------------------------
# Curva ROC comparativa
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 7))
for _, row in training_df.iterrows():
    model_name = row["modelo"]
    predictions_path = BASE_DIR / row["ficheiro_previsoes"]
    pred_df = pd.read_csv(predictions_path)
    y_true = pred_df["y_true"].astype(int)
    y_score = pred_df["y_score"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Aleatório (AUC=0.500)")
plt.title("Curvas ROC — Comparação de Modelos")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "curvas_roc_modelos.png", dpi=180)
plt.savefig(GRAPH_DIR / "curvas_roc_modelos.pdf")
plt.close()

# -------------------------------------------------------------------------
# Curva Precision-Recall comparativa
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 7))
for _, row in training_df.iterrows():
    model_name = row["modelo"]
    predictions_path = BASE_DIR / row["ficheiro_previsoes"]
    pred_df = pd.read_csv(predictions_path)
    y_true = pred_df["y_true"].astype(int)
    y_score = pred_df["y_score"]

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_score)
    plt.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={avg_precision:.3f})")

positive_rate = y_true.mean()
plt.axhline(positive_rate, linestyle="--", linewidth=1, label=f"Baseline classe positiva ({positive_rate:.3f})")
plt.title("Curvas Precision-Recall — Comparação de Modelos")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "curvas_precision_recall_modelos.png", dpi=180)
plt.savefig(GRAPH_DIR / "curvas_precision_recall_modelos.pdf")
plt.close()

print("\nGráficos guardados em:")
print(GRAPH_DIR / "curvas_roc_modelos.png")
print(GRAPH_DIR / "curvas_precision_recall_modelos.png")
print("\nEtapa 06 concluída com sucesso.")
