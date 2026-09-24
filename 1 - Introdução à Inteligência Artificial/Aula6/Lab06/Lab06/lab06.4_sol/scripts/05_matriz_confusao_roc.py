# 05_matriz_confusao_roc.py
# Lab 06.4 — Modelos Ensemble
# Etapa: Matriz de confusão, Curvas ROC e Precision-Recall
#
# Esta etapa visualiza o desempenho do melhor modelo e compara todos os modelos
# com curvas ROC e Precision-Recall.

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parents[1]
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
METRIC_DIR = BASE_DIR / "artefactos" / "metricas"
FIG_DIR = BASE_DIR / "artefactos" / "figuras"

FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.4 — ETAPA 5: MATRIZ DE CONFUSÃO E CURVAS")
print("=" * 80)

dados = joblib.load(PREP_DIR / "dados_processados.pkl")
y_test = dados["y_test"]
previsoes = joblib.load(MODEL_DIR / "previsoes_teste.pkl")
melhor = json.loads((METRIC_DIR / "melhor_modelo.json").read_text(encoding="utf-8"))
melhor_nome = melhor["modelo"]

print(f"\nMelhor modelo selecionado: {melhor_nome}")

y_pred = previsoes[melhor_nome]["y_pred"]

matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
matriz_percentual = matriz / matriz.sum() * 100

plt.figure(figsize=(7, 6))
anotacoes = [
    [f"{matriz[i, j]}\n({matriz_percentual[i, j]:.1f}%)" for j in range(2)]
    for i in range(2)
]
sns.heatmap(
    matriz,
    annot=anotacoes,
    fmt="",
    cmap="Blues",
    xticklabels=["Previsto 0", "Previsto 1"],
    yticklabels=["Real 0", "Real 1"],
)
plt.title(f"Matriz de Confusão — {melhor_nome}")
plt.ylabel("Classe real")
plt.xlabel("Classe prevista")
plt.tight_layout()
plt.savefig(FIG_DIR / "matriz_confusao_melhor_modelo.png", dpi=220)
plt.savefig(FIG_DIR / "matriz_confusao_melhor_modelo.pdf")
plt.close()

tn, fp, fn, tp = matriz.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

interpretação = f"""# Interpretação da Matriz de Confusão

## Melhor modelo

Modelo: `{melhor_nome}`

## Valores

- TN — Verdadeiros negativos: {tn}
- FP — Falsos positivos: {fp}
- FN — Falsos negativos: {fn}
- TP — Verdadeiros positivos: {tp}

## Indicadores derivados

- FPR — Taxa de falsos positivos: {fpr:.4f}
- FNR — Taxa de falsos negativos: {fnr:.4f}
- Specificity — capacidade de detetar corretamente voos sem incidente: {specificity:.4f}
- Recall — capacidade de detetar corretamente incidentes: {recall:.4f}

## Leitura no contexto do problema

Os falsos negativos são especialmente críticos: representam voos com incidente real que o
modelo classificou como seguros. Os falsos positivos podem gerar custos operacionais, mas
são geralmente menos graves do que ignorar um risco real.
"""
(METRIC_DIR / "interpretacao_matriz_confusao.md").write_text(interpretação, encoding="utf-8")

plt.figure(figsize=(8, 6))
for nome, valores in previsoes.items():
    y_score = valores["y_score"]
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr_curve, tpr_curve)
    plt.plot(fpr_curve, tpr_curve, label=f"{nome} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")
plt.title("Curvas ROC — Comparação dos Modelos")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos / Recall")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "curvas_roc_modelos.png", dpi=220)
plt.savefig(FIG_DIR / "curvas_roc_modelos.pdf")
plt.close()

plt.figure(figsize=(8, 6))
for nome, valores in previsoes.items():
    y_score = valores["y_score"]
    precision, recall_curve, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall_curve, precision)
    plt.plot(recall_curve, precision, label=f"{nome} (PR-AUC={pr_auc:.3f})")

baseline = y_test.mean()
plt.axhline(baseline, linestyle="--", label=f"Baseline classe positiva ({baseline:.3f})")
plt.title("Curvas Precision-Recall — Comparação dos Modelos")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "curvas_precision_recall_modelos.png", dpi=220)
plt.savefig(FIG_DIR / "curvas_precision_recall_modelos.pdf")
plt.close()

print("\nFiguras guardadas em:")
print(FIG_DIR)
print("\nEtapa 5 concluída com sucesso.")
