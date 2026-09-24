# 05_avaliacao_melhor_subconjunto.py
# Lab 08.2 — Algoritmos Genéticos
# Etapa 5: avaliação final do melhor subconjunto no conjunto de teste.

from pathlib import Path
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

BASE_DIR = Path(__file__).resolve().parent
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
GA_DIR = BASE_DIR / "artefactos" / "ga_features"
AVAL_DIR = BASE_DIR / "artefactos" / "avaliacao"
AVAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ETAPA 5 — Avaliação final do melhor subconjunto")
print("=" * 80)

X_train = pd.read_csv(PREP_DIR / "X_train_scaled.csv")
X_test = pd.read_csv(PREP_DIR / "X_test_scaled.csv")
y_train = pd.read_csv(PREP_DIR / "y_train.csv").iloc[:, 0].astype(int)
y_test = pd.read_csv(PREP_DIR / "y_test.csv").iloc[:, 0].astype(int)

with open(PREP_DIR / "feature_names.json", "r", encoding="utf-8") as f:
    feature_names = json.load(f)

with open(GA_DIR / "hof.pkl", "rb") as f:
    hof = pickle.load(f)

best_individual = hof[0]
selected_indices = [i for i, bit in enumerate(best_individual) if bit == 1]
selected_features = [feature_names[i] for i in selected_indices]

if len(selected_features) == 0:
    raise ValueError("O melhor indivíduo não selecionou features. Rever parâmetros do GA.")

print("Features selecionadas pelo GA:")
for feat in selected_features:
    print("-", feat)

def calcular_metricas(nome, modelo, Xtr, Xte):
    modelo.fit(Xtr, y_train)
    pred = modelo.predict(Xte)

    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(Xte)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
    else:
        roc_auc = np.nan

    return {
        "modelo": nome,
        "n_features": Xtr.shape[1],
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc,
    }, pred, modelo

base_model = LogisticRegression(
    max_iter=700,
    class_weight="balanced",
    solver="liblinear",
    random_state=42,
)

ga_model = LogisticRegression(
    max_iter=700,
    class_weight="balanced",
    solver="liblinear",
    random_state=42,
)

metricas_base, pred_base, trained_base = calcular_metricas("LogisticRegression_todas_features", base_model, X_train, X_test)
metricas_ga, pred_ga, trained_ga = calcular_metricas(
    "LogisticRegression_features_GA",
    ga_model,
    X_train.iloc[:, selected_indices],
    X_test.iloc[:, selected_indices],
)

metricas = pd.DataFrame([metricas_base, metricas_ga])
metricas = metricas.round(4)
metricas.to_csv(AVAL_DIR / "metricas_modelos.csv", index=False)
metricas.to_markdown(AVAL_DIR / "metricas_modelos.md", index=False)

print("\nMétricas finais:")
print(metricas)

report_text = classification_report(y_test, pred_ga, zero_division=0)
(AVAL_DIR / "classification_report_ga.txt").write_text(report_text, encoding="utf-8")

cm = confusion_matrix(y_test, pred_ga)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.title("Matriz de Confusão — Melhor Subconjunto GA")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(AVAL_DIR / "matriz_confusao_ga.png", dpi=170)
plt.savefig(AVAL_DIR / "matriz_confusao_ga.pdf")
plt.close()

best_features_data = {
    "selected_features": selected_features,
    "selected_indices": selected_indices,
    "n_features_selected": len(selected_features),
    "n_features_total": len(feature_names),
    "metricas_modelo_ga": metricas_ga,
}
with open(AVAL_DIR / "best_features.json", "w", encoding="utf-8") as f:
    json.dump(best_features_data, f, ensure_ascii=False, indent=2)

joblib.dump(trained_ga, AVAL_DIR / "modelo_logistic_regression_features_ga.pkl")
joblib.dump(trained_base, AVAL_DIR / "modelo_logistic_regression_todas_features.pkl")

md = f"""# Avaliação do Melhor Subconjunto

## Features selecionadas

{chr(10).join(f"- `{feat}`" for feat in selected_features)}

## Métricas

{metricas.to_markdown(index=False)}

## Matriz de Confusão

![Matriz de Confusão](matriz_confusao_ga.png)

## Relatório de classificação

```text
{report_text}
```
"""
(AVAL_DIR / "avaliacao_melhor_subconjunto.md").write_text(md, encoding="utf-8")

print("\nArtefactos guardados em:", AVAL_DIR)
print("Etapa 5 concluída com sucesso.")
