# -*- coding: utf-8 -*-
"""
Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos
Etapa 05: Matriz de Confusão

Objetivo:
- Selecionar o melhor modelo já identificado na etapa de avaliação.
- Gerar matriz de confusão com contagens e percentagens.
- Interpretar TN, FP, FN e TP no contexto de segurança de voo.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("=" * 80)
print("ETAPA 05 — MATRIZ DE CONFUSÃO")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / "artefactos" / "previsoes"
METRICS_DIR = BASE_DIR / "artefactos" / "metricas"
GRAPH_DIR = BASE_DIR / "artefactos" / "graficos"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

best_path = METRICS_DIR / "melhor_modelo.json"
if not best_path.exists():
    raise FileNotFoundError(f"Ficheiro do melhor modelo não encontrado: {best_path}")

best_info = json.loads(best_path.read_text(encoding="utf-8"))
best_model = best_info["melhor_modelo"]
predictions_path = PRED_DIR / f"previsoes_{best_model}.csv"
if not predictions_path.exists():
    raise FileNotFoundError(f"Previsões do melhor modelo em falta: {predictions_path}")

pred_df = pd.read_csv(predictions_path)
y_true = pred_df["y_true"].astype(int)
y_pred = pred_df["y_pred"].astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
cm_percent_total = cm / cm.sum() * 100
cm_percent_by_true = cm / cm.sum(axis=1, keepdims=True) * 100

specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

print(f"\nMelhor modelo: {best_model}")
print("\nMatriz de confusão [linhas = real, colunas = previsto]:")
print(cm)
print("\nInterpretação binária:")
print(f"TN — Sem incidente previsto como sem incidente: {tn}")
print(f"FP — Sem incidente previsto como incidente: {fp}")
print(f"FN — Incidente real previsto como sem incidente: {fn}")
print(f"TP — Incidente real previsto como incidente: {tp}")
print(f"Recall / Sensibilidade: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"FNR: {fnr:.4f}")

annotations = np.empty_like(cm).astype(object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annotations[i, j] = (
            f"{cm[i, j]}\n"
            f"{cm_percent_total[i, j]:.1f}% do total\n"
            f"{cm_percent_by_true[i, j]:.1f}% da classe real"
        )

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=annotations,
    fmt="",
    cmap="Blues",
    xticklabels=["Previsto 0", "Previsto 1"],
    yticklabels=["Real 0", "Real 1"],
    cbar=False,
)
plt.title(f"Matriz de Confusão — {best_model}")
plt.xlabel("Classe prevista")
plt.ylabel("Classe real")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "matriz_confusao_melhor_modelo.png", dpi=180)
plt.savefig(GRAPH_DIR / "matriz_confusao_melhor_modelo.pdf")
plt.close()

summary = {
    "melhor_modelo": best_model,
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "recall": round(float(recall), 6),
    "specificity": round(float(specificity), 6),
    "fpr": round(float(fpr), 6),
    "fnr": round(float(fnr), 6),
    "interpretacao": {
        "falsos_negativos": "Incidentes reais que o modelo classificou como seguros. Em segurança, este é o erro mais crítico.",
        "falsos_positivos": "Voos seguros classificados como risco. Criam alertas desnecessários, mas tendem a ser menos graves do que falsos negativos.",
    },
}
(METRICS_DIR / "matriz_confusao_melhor_modelo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nGráficos guardados em:")
print(GRAPH_DIR / "matriz_confusao_melhor_modelo.png")
print(GRAPH_DIR / "matriz_confusao_melhor_modelo.pdf")
print("\nEtapa 05 concluída com sucesso.")
