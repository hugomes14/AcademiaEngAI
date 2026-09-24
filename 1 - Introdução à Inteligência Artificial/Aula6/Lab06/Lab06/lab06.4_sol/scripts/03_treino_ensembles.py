# 03_treino_ensembles.py
# Lab 06.4 — Modelos Ensemble
# Etapa: Treino de modelos
#
# Modelos treinados:
# - DecisionTreeClassifier como baseline.
# - RandomForestClassifier como bagging.
# - XGBClassifier como boosting, se xgboost estiver instalado.
# - GradientBoostingClassifier como alternativa caso xgboost não esteja instalado.
# - CatBoostClassifier, se catboost estiver instalado.
#
# Esta etapa não calcula métricas finais. Apenas treina, prevê e guarda artefactos.

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
METRIC_DIR = BASE_DIR / "artefactos" / "metricas"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.4 — ETAPA 3: TREINO DE MODELOS ENSEMBLE")
print("=" * 80)

dados = joblib.load(PREP_DIR / "dados_processados.pkl")
X_train = dados["X_train"]
X_test = dados["X_test"]
y_train = dados["y_train"]
y_test = dados["y_test"]

print(f"\nTreino: {X_train.shape}")
print(f"Teste:  {X_test.shape}")

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale_pos_weight = n_neg / max(n_pos, 1)

print("\nDistribuição no treino:")
print(y_train.value_counts().sort_index())
print(f"\nPeso relativo para classe positiva usado no XGBoost: {scale_pos_weight:.2f}")

modelos = {}

modelos["decision_tree"] = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
)

modelos["random_forest"] = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    oob_score=True,
    bootstrap=True,
    n_jobs=1,
)

# XGBoost com validação cruzada simples para escolher n_estimators.
cv_resultados = []
try:
    from xgboost import XGBClassifier

    print("\nXGBoost disponível. A testar n_estimators com validação cruzada...")
    candidatos = [50, 100]
    melhor_score = -1
    melhor_n = candidatos[0]

    for n_estimators in candidatos:
        modelo_cv = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            modelo_cv,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=1,
        )

        media = float(np.mean(scores))
        desvio = float(np.std(scores))
        cv_resultados.append({
            "modelo": "xgboost",
            "n_estimators": n_estimators,
            "f1_cv_medio": media,
            "f1_cv_desvio": desvio,
        })
        print(f"  n_estimators={n_estimators}: F1 CV={media:.4f} ± {desvio:.4f}")

        if media > melhor_score:
            melhor_score = media
            melhor_n = n_estimators

    print(f"\nMelhor n_estimators para XGBoost: {melhor_n}")
    modelos["xgboost"] = XGBClassifier(
        n_estimators=melhor_n,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
    )

except Exception as erro:
    print("\nXGBoost indisponível ou falhou. A usar GradientBoostingClassifier como alternativa.")
    print(f"Motivo: {erro}")
    modelos["gradient_boosting_fallback"] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

# CatBoost é opcional e pode não existir no ambiente do formando.
try:
    from catboost import CatBoostClassifier

    print("\nCatBoost disponível. A incluir modelo opcional.")
    modelos["catboost"] = CatBoostClassifier(
        iterations=80,
        depth=4,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=42,
        verbose=False,
        class_weights=[1.0, scale_pos_weight],
    )

except Exception as erro:
    print("\nCatBoost indisponível. Esta etapa opcional será ignorada.")
    print(f"Motivo: {erro}")

if cv_resultados:
    pd.DataFrame(cv_resultados).to_csv(METRIC_DIR / "validacao_cruzada_xgboost.csv", index=False)

previsoes = {}
tempos = []

for nome, modelo in modelos.items():
    print("\n" + "-" * 80)
    print(f"A treinar modelo: {nome}")
    inicio = time.perf_counter()

    modelo.fit(X_train, y_train)

    duracao = time.perf_counter() - inicio
    tempos.append({"modelo": nome, "tempo_treino_segundos": duracao})

    print(f"Tempo de treino: {duracao:.3f} segundos")

    y_pred = modelo.predict(X_test)

    if hasattr(modelo, "predict_proba"):
        y_score = modelo.predict_proba(X_test)[:, 1]
    elif hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(X_test)
        y_score = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        y_score = y_pred.astype(float)

    previsoes[nome] = {
        "y_pred": pd.Series(y_pred, index=y_test.index),
        "y_score": pd.Series(y_score, index=y_test.index),
    }

    joblib.dump(modelo, MODEL_DIR / f"{nome}.pkl")

    if nome == "random_forest":
        print(f"OOB score do Random Forest: {modelo.oob_score_:.4f}")
        (METRIC_DIR / "random_forest_oob_score.txt").write_text(
            f"OOB score: {modelo.oob_score_:.6f}\n",
            encoding="utf-8",
        )

joblib.dump(previsoes, MODEL_DIR / "previsoes_teste.pkl")
pd.DataFrame(tempos).to_csv(METRIC_DIR / "tempos_treino.csv", index=False)

resumo_modelos = {
    "modelos_treinados": list(modelos.keys()),
    "scale_pos_weight": scale_pos_weight,
    "observacao": "Métricas finais são calculadas em 04_avaliacao_metricas.py",
}
(MODEL_DIR / "resumo_modelos.json").write_text(
    json.dumps(resumo_modelos, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

notas = """# Notas de Treino

## Árvore de Decisão

Serve como baseline. Sem limite de profundidade, pode ajustar-se demasiado ao treino e generalizar mal.

## Random Forest

Usa bagging: treina muitas árvores em amostras diferentes. Isto reduz variância face a uma árvore isolada.
Foi usado `oob_score=True` para obter uma estimativa interna de desempenho.

## XGBoost

Usa boosting: treina modelos sequencialmente, cada um a tentar corrigir erros dos anteriores.
Foi usada validação cruzada para escolher `n_estimators`.

## CatBoost

É opcional. Quando disponível, entra como comparação adicional.
"""
(MODEL_DIR / "notas_treino.md").write_text(notas, encoding="utf-8")

print("\nModelos e previsões guardados em:")
print(MODEL_DIR)
print("\nEtapa 3 concluída com sucesso.")
