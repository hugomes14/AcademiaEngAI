# -*- coding: utf-8 -*-
"""
Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos
Etapa 03: Treino de Modelos

Objetivo:
- Carregar dados processados.
- Treinar os modelos pedidos no laboratório.
- Guardar modelos, previsões e tempos de treino.
- Não calcular métricas nesta etapa, para manter responsabilidades separadas.
"""

from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

print("=" * 80)
print("ETAPA 03 — TREINO DE MODELOS")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "artefactos" / "dados"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
PRED_DIR = BASE_DIR / "artefactos" / "previsoes"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "incidente_reportado"

required_files = [
    DATA_DIR / "X_train_scaled.csv",
    DATA_DIR / "X_test_scaled.csv",
    DATA_DIR / "X_train_unscaled.csv",
    DATA_DIR / "X_test_unscaled.csv",
    DATA_DIR / "y_train.csv",
    DATA_DIR / "y_test.csv",
]
for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro obrigatório em falta: {path}")

X_train_scaled = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
X_test_scaled = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
X_train_unscaled = pd.read_csv(DATA_DIR / "X_train_unscaled.csv")
X_test_unscaled = pd.read_csv(DATA_DIR / "X_test_unscaled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET]
y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET]

print("\nDimensões carregadas:")
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"X_test_scaled:  {X_test_scaled.shape}")
print(f"y_train:        {y_train.shape}")
print(f"y_test:         {y_test.shape}")

# Modelos principais pedidos pelo enunciado.
# Inclui versões com class_weight='balanced' para discutir desbalanceamento.
model_specs = [
    {
        "name": "Regressao_Logistica",
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "scaled": True,
        "note": "Baseline linear interpretável.",
    },
    {
        "name": "Regressao_Logistica_Balanced",
        "model": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "scaled": True,
        "note": "Versão com pesos de classe para compensar desbalanceamento.",
    },
    {
        "name": "KNN",
        "model": KNeighborsClassifier(n_neighbors=7),
        "scaled": True,
        "note": "Sensível à escala; usa distâncias entre exemplos.",
    },
    {
        "name": "SVM_Linear",
        "model": SVC(kernel="linear", probability=True, random_state=42),
        "scaled": True,
        "note": "Margem linear; sensível à escala.",
    },
    {
        "name": "SVM_Linear_Balanced",
        "model": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
        "scaled": True,
        "note": "SVM linear com pesos de classe.",
    },
    {
        "name": "SVM_RBF",
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "scaled": True,
        "note": "Kernel não linear; pode captar fronteiras mais complexas.",
    },
    {
        "name": "SVM_RBF_Balanced",
        "model": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "scaled": True,
        "note": "Kernel RBF com pesos de classe para a classe minoritária.",
    },
    {
        "name": "Naive_Bayes_Gaussiano",
        "model": GaussianNB(),
        "scaled": True,
        "note": "Rápido e simples; assume distribuição gaussiana por feature.",
    },
    {
        "name": "KNN_Sem_Escala",
        "model": KNeighborsClassifier(n_neighbors=7),
        "scaled": False,
        "note": "Modelo extra para demonstrar o impacto de não escalar variáveis.",
    },
    {
        "name": "SVM_RBF_Sem_Escala",
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "scaled": False,
        "note": "Modelo extra para demonstrar o impacto de não escalar variáveis.",
    },
]

training_records = []

for spec in model_specs:
    name = spec["name"]
    model = spec["model"]
    uses_scaled_data = bool(spec["scaled"])

    print("\n" + "-" * 80)
    print(f"A treinar: {name}")
    print(f"Nota: {spec['note']}")

    if uses_scaled_data:
        X_train_current = X_train_scaled
        X_test_current = X_test_scaled
        data_version = "scaled"
    else:
        X_train_current = X_train_unscaled
        X_test_current = X_test_unscaled
        data_version = "unscaled"

    start_time = time.perf_counter()
    model.fit(X_train_current, y_train)
    elapsed_seconds = time.perf_counter() - start_time

    y_pred = model.predict(X_test_current)

    # Nem todos os modelos expõem predict_proba. Neste laboratório, todos os modelos escolhidos expõem,
    # porque SVC foi criado com probability=True. Ainda assim, mantemos fallback com decision_function.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_current)[:, 1]
        score_type = "predict_proba"
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_current)
        score_type = "decision_function"
    else:
        y_score = y_pred
        score_type = "hard_prediction"

    model_path = MODEL_DIR / f"{name}.pkl"
    predictions_path = PRED_DIR / f"previsoes_{name}.csv"
    joblib.dump(model, model_path)

    pred_df = pd.DataFrame({
        "y_true": y_test.astype(int),
        "y_pred": y_pred.astype(int),
        "y_score": y_score,
    })
    pred_df.to_csv(predictions_path, index=False, encoding="utf-8")

    record = {
        "modelo": name,
        "ficheiro_modelo": str(model_path.relative_to(BASE_DIR)),
        "ficheiro_previsoes": str(predictions_path.relative_to(BASE_DIR)),
        "dados_usados": data_version,
        "score_type": score_type,
        "tempo_treino_segundos": round(elapsed_seconds, 6),
        "nota": spec["note"],
    }
    training_records.append(record)

    print(f"Modelo guardado em: {model_path}")
    print(f"Previsões guardadas em: {predictions_path}")
    print(f"Tempo de treino: {elapsed_seconds:.4f} segundos")

training_df = pd.DataFrame(training_records)
training_df.to_csv(MODEL_DIR / "registo_treino_modelos.csv", index=False, encoding="utf-8")
(MODEL_DIR / "registo_treino_modelos.json").write_text(
    json.dumps(training_records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nResumo do treino:")
print(training_df[["modelo", "dados_usados", "tempo_treino_segundos"]].to_string(index=False))
print("\nEtapa 03 concluída com sucesso.")
