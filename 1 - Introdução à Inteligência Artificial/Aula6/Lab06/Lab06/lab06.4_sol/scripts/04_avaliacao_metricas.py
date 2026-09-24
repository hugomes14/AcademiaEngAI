# 04_avaliacao_metricas.py
# Lab 06.4 — Modelos Ensemble
# Etapa: Avaliação e métricas
#
# Como o dataset é desbalanceado, são calculadas várias métricas:
# - Accuracy: útil, mas perigosa se usada isoladamente.
# - Precision: entre alertas positivos, quantos são corretos.
# - Recall: entre incidentes reais, quantos foram detetados.
# - F1: equilíbrio entre Precision e Recall.
# - ROC-AUC: separação global entre classes.
# - PR-AUC: muito útil em classes raras.
# - Specificity: capacidade de identificar corretamente a classe negativa.

from pathlib import Path
import json

import joblib
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

BASE_DIR = Path(__file__).resolve().parents[1]
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
METRIC_DIR = BASE_DIR / "artefactos" / "metricas"

METRIC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.4 — ETAPA 4: AVALIAÇÃO E MÉTRICAS")
print("=" * 80)

dados = joblib.load(PREP_DIR / "dados_processados.pkl")
y_test = dados["y_test"]
previsoes = joblib.load(MODEL_DIR / "previsoes_teste.pkl")

linhas = []

for nome, valores in previsoes.items():
    y_pred = valores["y_pred"]
    y_score = valores["y_score"]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    try:
        roc_auc = roc_auc_score(y_test, y_score)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_test, y_score)
    except Exception:
        pr_auc = float("nan")

    linhas.append({
        "modelo": nome,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

metricas = pd.DataFrame(linhas)

# Critério principal: F1. Critério secundário: ROC-AUC.
metricas_ordenadas = metricas.sort_values(
    by=["f1", "roc_auc", "recall"],
    ascending=[False, False, False],
).reset_index(drop=True)

melhor = metricas_ordenadas.iloc[0].to_dict()

print("\nTabela de métricas:")
print(metricas_ordenadas.round(4))

metricas_ordenadas.to_csv(METRIC_DIR / "metricas_modelos.csv", index=False)
metricas_ordenadas.round(4).to_markdown(METRIC_DIR / "metricas_modelos.md", index=False)

(METRIC_DIR / "melhor_modelo.json").write_text(
    json.dumps(melhor, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

discussao = f"""# Discussão das Métricas

## Melhor modelo

O melhor modelo pelo critério principal **F1-Score** foi:

- Modelo: `{melhor["modelo"]}`
- F1: {melhor["f1"]:.4f}
- ROC-AUC: {melhor["roc_auc"]:.4f}
- Recall: {melhor["recall"]:.4f}
- Precision: {melhor["precision"]:.4f}

## Por que a Accuracy não chega?

Como a classe positiva representa uma minoria, um modelo que prevê sempre "sem incidente"
pode alcançar uma Accuracy elevada. No entanto, esse modelo falha precisamente no que interessa:
detetar incidentes.

## Custo de erros

- **Falso Negativo (FN):** o modelo prevê ausência de incidente, mas ocorre um incidente. É o erro mais crítico em segurança.
- **Falso Positivo (FP):** o modelo prevê risco, mas não ocorre incidente. Pode gerar inspeções adicionais e custos operacionais.

## Métrica principal

O F1-Score foi escolhido porque equilibra Precision e Recall. Num contexto de segurança,
também se deve observar o Recall, porque falhar incidentes reais pode ter custo elevado.
"""
(METRIC_DIR / "discussao_metricas.md").write_text(discussao, encoding="utf-8")

print("\nMelhor modelo:")
print(json.dumps(melhor, ensure_ascii=False, indent=2))

print("\nArtefactos de métricas guardados em:")
print(METRIC_DIR)
print("\nEtapa 4 concluída com sucesso.")
