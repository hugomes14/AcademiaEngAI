# -*- coding: utf-8 -*-
"""
Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos
Etapa 04: Avaliação e Métricas

Objetivo:
- Calcular métricas de classificação binária.
- Comparar modelos de forma tabular.
- Destacar o melhor modelo por F1-Score, com ROC-AUC como critério de desempate.
- Guardar resultados em CSV e Markdown.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

print("=" * 80)
print("ETAPA 04 — AVALIAÇÃO E MÉTRICAS")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = BASE_DIR / "artefactos" / "previsoes"
METRICS_DIR = BASE_DIR / "artefactos" / "metricas"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

training_log_path = MODEL_DIR / "registo_treino_modelos.csv"
if not training_log_path.exists():
    raise FileNotFoundError(f"Registo de treino não encontrado: {training_log_path}")

training_df = pd.read_csv(training_log_path)
records = []

for _, row in training_df.iterrows():
    model_name = row["modelo"]
    predictions_path = BASE_DIR / row["ficheiro_previsoes"]
    if not predictions_path.exists():
        raise FileNotFoundError(f"Previsões em falta para {model_name}: {predictions_path}")

    pred_df = pd.read_csv(predictions_path)
    y_true = pred_df["y_true"].astype(int)
    y_pred = pred_df["y_pred"].astype(int)
    y_score = pred_df["y_score"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan

    records.append({
        "modelo": model_name,
        "dados_usados": row.get("dados_usados", ""),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "tempo_treino_segundos": row.get("tempo_treino_segundos", np.nan),
    })

metrics_df = pd.DataFrame(records)
metrics_df = metrics_df.sort_values(by=["f1", "roc_auc", "recall"], ascending=[False, False, False]).reset_index(drop=True)

rounded_df = metrics_df.copy()
metric_cols = [
    "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
    "fpr", "fnr", "tempo_treino_segundos",
]
for col in metric_cols:
    if col in rounded_df.columns:
        rounded_df[col] = rounded_df[col].astype(float).round(4)

metrics_csv = METRICS_DIR / "metricas_classificacao.csv"
metrics_md = METRICS_DIR / "metricas_classificacao.md"
metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")
metrics_md.write_text(rounded_df.to_markdown(index=False), encoding="utf-8")

best_row = metrics_df.iloc[0].to_dict()
best_model_info = {
    "criterio_principal": "f1",
    "criterio_desempate": "roc_auc",
    "melhor_modelo": best_row["modelo"],
    "dados_usados": best_row.get("dados_usados", ""),
    "f1": round(float(best_row["f1"]), 6),
    "roc_auc": round(float(best_row["roc_auc"]), 6) if not pd.isna(best_row["roc_auc"]) else None,
    "recall": round(float(best_row["recall"]), 6),
    "precision": round(float(best_row["precision"]), 6),
}
(METRICS_DIR / "melhor_modelo.json").write_text(json.dumps(best_model_info, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nTabela comparativa de métricas:")
print(rounded_df.to_string(index=False))

print("\nMelhor modelo selecionado:")
print(json.dumps(best_model_info, ensure_ascii=False, indent=2))

print("\nNotas de interpretação:")
print("- Accuracy pode ser enganadora porque a classe sem incidente é dominante.")
print("- Recall mede a capacidade de encontrar incidentes reais; é crítico em segurança.")
print("- Precision indica quantos alertas positivos estavam corretos.")
print("- F1 equilibra Precision e Recall, útil quando existe desbalanceamento.")
print("- Specificity mede a capacidade de reconhecer voos sem incidente.")
print("- ROC-AUC resume separabilidade global; PR-AUC é útil quando a classe positiva é rara.")

print("\nFicheiros guardados:")
print(metrics_csv)
print(metrics_md)
print("\nEtapa 04 concluída com sucesso.")
