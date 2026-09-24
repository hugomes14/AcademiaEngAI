# Lab 06.3 — Séries Temporais
# Script 04 — Avaliação e Métricas
# Objetivo: calcular MAE, MSE, RMSE e MAPE para todos os modelos e escolher
# o melhor por série com base no menor RMSE.

from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
PREV_DIR = ARTEFACTOS_DIR / "previsoes"
TAB_DIR = ARTEFACTOS_DIR / "tabelas"

TAB_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SCRIPT 04 — AVALIAÇÃO E MÉTRICAS")
print("=" * 90)

linhas_resultados = []
melhores_modelos = {}

for nome in ["voltagem", "missoes"]:
    print("\n" + "-" * 90)
    print(f"[AVALIAÇÃO] SÉRIE: {nome.upper()}")
    print("-" * 90)

    previsoes = pd.read_csv(PREV_DIR / f"{nome}_previsoes.csv", parse_dates=["data_hora"])
    previsoes = previsoes.set_index("data_hora")

    y_real = previsoes["real"].astype(float)

    modelos = [c for c in previsoes.columns if c != "real"]

    for modelo in modelos:
        y_pred = previsoes[modelo].astype(float)

        # Remover valores inválidos, se algum modelo tiver falhado parcialmente.
        mascara = np.isfinite(y_real) & np.isfinite(y_pred)
        y_true_valid = y_real[mascara]
        y_pred_valid = y_pred[mascara]

        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        mse = mean_squared_error(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_valid, y_pred_valid)

        # MAPE: evitar divisão por zero.
        denominador = np.where(np.abs(y_true_valid) < 1e-8, np.nan, np.abs(y_true_valid))
        mape = np.nanmean(np.abs((y_true_valid - y_pred_valid) / denominador)) * 100

        # sMAPE: métrica complementar mais estável quando valores reais se aproximam de zero.
        smape = np.mean(
            2 * np.abs(y_pred_valid - y_true_valid)
            / (np.abs(y_true_valid) + np.abs(y_pred_valid) + 1e-8)
        ) * 100

        linhas_resultados.append({
            "serie": nome,
            "modelo": modelo,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE_percent": mape,
            "sMAPE_percent": smape,
            "R2": r2,
            "n_observacoes": int(len(y_true_valid)),
        })

    resultados_serie = pd.DataFrame([r for r in linhas_resultados if r["serie"] == nome])
    resultados_serie = resultados_serie.sort_values("RMSE")
    melhor = resultados_serie.iloc[0]["modelo"]
    melhores_modelos[nome] = {
        "modelo": melhor,
        "criterio": "menor RMSE",
        "RMSE": float(resultados_serie.iloc[0]["RMSE"]),
    }

    print(resultados_serie[["modelo", "MAE", "RMSE", "MAPE_percent", "sMAPE_percent", "R2"]].round(4))
    print(f"[MELHOR] {nome}: {melhor} por menor RMSE.")

resultados = pd.DataFrame(linhas_resultados)

# Formatação para leitura.
resultados_formatados = resultados.copy()
for col in ["MAE", "MSE", "RMSE", "MAPE_percent", "sMAPE_percent", "R2"]:
    resultados_formatados[col] = resultados_formatados[col].map(lambda x: f"{x:.4f}")

resultados.to_csv(TAB_DIR / "metricas_modelos.csv", index=False)
resultados_formatados.to_markdown(TAB_DIR / "metricas_modelos.md", index=False)

with open(TAB_DIR / "melhores_modelos.json", "w", encoding="utf-8") as f:
    json.dump(melhores_modelos, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 90)
print("[OK] Métricas guardadas em artefactos/tabelas/.")
print("[INFO] O baseline de persistência deve ser usado como referência mínima.")
print("[INFO] Um modelo útil deve reduzir MAE/RMSE face ao baseline.")
print("=" * 90)
