# -*- coding: utf-8 -*-
"""
Lab 06.5 - Treino de Regressão Linear Múltipla com sklearn.

Este script treina um modelo clássico de regressão linear sobre exatamente os mesmos
conjuntos processados usados pelo modelo PyTorch. Assim, a comparação é justa.
"""

from pathlib import Path
import pickle
import time

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

RAIZ = Path(__file__).resolve().parent
DIR_PREP = RAIZ / "artefactos" / "preprocessamento"
DIR_MODELOS = RAIZ / "artefactos" / "modelos"
DIR_PREV = RAIZ / "artefactos" / "previsoes"
DIR_METRICAS = RAIZ / "artefactos" / "metricas"

DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_PREV.mkdir(parents=True, exist_ok=True)
DIR_METRICAS.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.5 - TREINO SKLEARN: REGRESSÃO LINEAR MÚLTIPLA")
print("=" * 80)

with open(DIR_PREP / "dados_processados.pkl", "rb") as f:
    dados = pickle.load(f)

X_train = dados["X_train_proc"]
X_val = dados["X_val_proc"]
X_test = dados["X_test_proc"]
y_train = dados["y_train_array"]
y_val = dados["y_val_array"]
y_test = dados["y_test_array"]
feature_names = dados["feature_names"]

inicio = time.time()
modelo = LinearRegression()
modelo.fit(X_train, y_train)
tempo = time.time() - inicio

pred_train = modelo.predict(X_train)
pred_val = modelo.predict(X_val)
pred_test = modelo.predict(X_test)

joblib.dump(modelo, DIR_MODELOS / "modelo_sklearn_regressao_linear.pkl")

pd.DataFrame({"y_real": y_test, "y_pred_sklearn": pred_test}).to_csv(
    DIR_PREV / "previsoes_sklearn_teste.csv", index=False
)
pd.DataFrame({"y_real": y_val, "y_pred_sklearn": pred_val}).to_csv(
    DIR_PREV / "previsoes_sklearn_validacao.csv", index=False
)

coef_df = pd.DataFrame({"feature": feature_names, "peso_sklearn": modelo.coef_.ravel()})
coef_df.loc[len(coef_df)] = ["bias/intercept", float(modelo.intercept_)]
coef_df.to_csv(DIR_METRICAS / "coeficientes_sklearn.csv", index=False)

pd.DataFrame(
    [{"modelo": "Regressão Linear sklearn", "tempo_treino_segundos": tempo}]
).to_csv(DIR_METRICAS / "tempo_treino_sklearn.csv", index=False)

print(f"Tempo de treino sklearn: {tempo:.6f} segundos")
print("\nCoeficientes do modelo sklearn:")
print(coef_df)
print("\nModelo sklearn, previsões e coeficientes guardados com sucesso.")
