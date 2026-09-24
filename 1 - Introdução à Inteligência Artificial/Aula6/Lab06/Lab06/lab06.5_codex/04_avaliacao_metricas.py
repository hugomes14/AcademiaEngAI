# -*- coding: utf-8 -*-
"""
Lab 06.5 - Avaliação dos modelos PyTorch e sklearn.

Métricas calculadas:
- MAE: erro absoluto médio, em minutos;
- MSE: erro quadrático médio;
- RMSE: raiz do MSE, em minutos;
- R²: proporção da variância explicada pelo modelo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RAIZ = Path(__file__).resolve().parent
DIR_PREV = RAIZ / "artefactos" / "previsoes"
DIR_METRICAS = RAIZ / "artefactos" / "metricas"
DIR_METRICAS.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.5 - AVALIAÇÃO E COMPARAÇÃO DE MÉTRICAS")
print("=" * 80)

prev_pytorch = pd.read_csv(DIR_PREV / "previsoes_pytorch_teste.csv")
prev_sklearn = pd.read_csv(DIR_PREV / "previsoes_sklearn_teste.csv")

def calcular_metricas(nome_modelo, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_real, y_pred)
    return {
        "modelo": nome_modelo,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

metricas = [
    calcular_metricas(
        "PyTorch - 1 neurónio linear",
        prev_pytorch["y_real"],
        prev_pytorch["y_pred_pytorch"],
    ),
    calcular_metricas(
        "sklearn - Regressão Linear",
        prev_sklearn["y_real"],
        prev_sklearn["y_pred_sklearn"],
    ),
]

metricas_df = pd.DataFrame(metricas)
metricas_df = metricas_df.sort_values("RMSE", ascending=True).reset_index(drop=True)
metricas_df.to_csv(DIR_METRICAS / "metricas_modelos.csv", index=False)

try:
    metricas_df.to_markdown(DIR_METRICAS / "metricas_modelos.md", index=False, floatfmt=".4f")
except Exception:
    (DIR_METRICAS / "metricas_modelos.md").write_text(metricas_df.to_string(index=False), encoding="utf-8")

print("\nTabela de métricas no conjunto de teste:")
print(metricas_df.to_string(index=False))

# Comparação dos coeficientes.
coef_pytorch = pd.read_csv(DIR_METRICAS / "coeficientes_pytorch.csv")
coef_sklearn = pd.read_csv(DIR_METRICAS / "coeficientes_sklearn.csv")
coef_comp = coef_pytorch.merge(coef_sklearn, on="feature", how="outer")
coef_comp["diferenca_absoluta"] = (coef_comp["peso_pytorch"] - coef_comp["peso_sklearn"]).abs()
coef_comp.to_csv(DIR_METRICAS / "comparacao_coeficientes.csv", index=False)

try:
    coef_comp.to_markdown(DIR_METRICAS / "comparacao_coeficientes.md", index=False, floatfmt=".6f")
except Exception:
    (DIR_METRICAS / "comparacao_coeficientes.md").write_text(coef_comp.to_string(index=False), encoding="utf-8")

print("\nComparação de coeficientes em espaço de features processadas/escaladas:")
print(coef_comp.to_string(index=False))

melhor = metricas_df.iloc[0]
print("\nMelhor modelo pelo critério RMSE:")
print(f"- {melhor['modelo']} | RMSE={melhor['RMSE']:.4f} | R²={melhor['R2']:.4f}")

print("\nNota de interpretação:")
print(
    "Como o modelo PyTorch tem apenas uma camada linear sem ativação, ele representa "
    "a mesma família matemática da regressão linear. Diferenças residuais podem surgir "
    "por inicialização aleatória, número de épocas, learning rate e otimização iterativa."
)
