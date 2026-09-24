# Lab 06.3 — Séries Temporais
# Script 03 — Treino de Modelos
# Objetivo: treinar baseline de persistência, modelos ML com lags/calendário,
# modelos simples de suavização exponencial e Prophet se estiver disponível.
#
# Este script NÃO calcula métricas. Apenas treina modelos, gera previsões e guarda artefactos.

from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


BASE_DIR = Path(__file__).resolve().parents[1]
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
OBJ_DIR = ARTEFACTOS_DIR / "objetos"
MODELOS_DIR = ARTEFACTOS_DIR / "modelos"
PREV_DIR = ARTEFACTOS_DIR / "previsoes"

MODELOS_DIR.mkdir(parents=True, exist_ok=True)
PREV_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SCRIPT 03 — TREINO DE MODELOS")
print("=" * 90)

resumo_tempos = []

for nome in ["voltagem", "missoes"]:
    print("\n" + "-" * 90)
    print(f"[TREINO] SÉRIE: {nome.upper()}")
    print("-" * 90)

    pacote = joblib.load(OBJ_DIR / f"{nome}_dados_processados.pkl")
    X_train_scaled = pacote["X_train_scaled"]
    X_test_scaled = pacote["X_test_scaled"]
    y_train = pacote["y_train"]
    y_test = pacote["y_test"]

    previsoes = pd.DataFrame(index=y_test.index)
    previsoes["real"] = y_test.values

    modelos = {}

    # ---------------------------------------------------------------------
    # Baseline Naive / Persistência.
    # Para cada instante do teste, prevê-se o valor observado imediatamente antes.
    # A primeira previsão usa o último valor do treino.
    # Este baseline é essencial: modelos mais complexos só são úteis se o superarem.
    # ---------------------------------------------------------------------
    inicio = time.perf_counter()
    valores_previos = pd.concat([y_train.tail(1), y_test.iloc[:-1]])
    previsoes["Baseline_Persistencia"] = valores_previos.values
    resumo_tempos.append({"serie": nome, "modelo": "Baseline_Persistencia", "tempo_segundos": time.perf_counter() - inicio})
    print("[OK] Baseline de persistência criado.")

    # ---------------------------------------------------------------------
    # Modelos ML com features de lag e calendário.
    # Regressão Linear: simples e interpretável.
    # Ridge: versão regularizada, útil quando existem muitas features correlacionadas.
    # Random Forest: não linear, capta interações, mas é menos interpretável.
    # ---------------------------------------------------------------------
    modelos_a_treinar = {
        "Regressao_Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Random_Forest": RandomForestRegressor(
            n_estimators=35,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        ),
    }

    for nome_modelo, modelo in modelos_a_treinar.items():
        inicio = time.perf_counter()
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)

        modelos[nome_modelo] = modelo
        previsoes[nome_modelo] = y_pred

        tempo = time.perf_counter() - inicio
        resumo_tempos.append({"serie": nome, "modelo": nome_modelo, "tempo_segundos": tempo})
        print(f"[OK] {nome_modelo} treinado em {tempo:.2f}s.")

    # ---------------------------------------------------------------------
    # Suavização Exponencial simples.
    # Modelo clássico leve. Serve como alternativa estatística ao baseline.
    # ---------------------------------------------------------------------
    try:
        inicio = time.perf_counter()
        ses = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(optimized=True)
        y_pred_ses = ses.forecast(len(y_test))
        previsoes["Suavizacao_Exponencial"] = np.asarray(y_pred_ses)

        modelos["Suavizacao_Exponencial"] = ses
        tempo = time.perf_counter() - inicio
        resumo_tempos.append({"serie": nome, "modelo": "Suavizacao_Exponencial", "tempo_segundos": tempo})
        print(f"[OK] Suavização Exponencial treinada em {tempo:.2f}s.")
    except Exception as erro:
        print(f"[AVISO] Suavização Exponencial não foi treinada: {erro}")

    # ---------------------------------------------------------------------
    # Prophet opcional.
    # Não é obrigatório para este pacote, porque pode exigir instalação adicional.
    # Se estiver instalado, o script usa-o.
    # ---------------------------------------------------------------------
    try:
        from prophet import Prophet

        inicio = time.perf_counter()
        df_prophet_train = pd.DataFrame({
            "ds": y_train.index,
            "y": y_train.values,
        })
        df_prophet_future = pd.DataFrame({
            "ds": y_test.index,
        })

        modelo_prophet = Prophet(
            yearly_seasonality=(nome == "missoes"),
            weekly_seasonality=True,
            daily_seasonality=(nome == "voltagem"),
            interval_width=0.95,
        )
        modelo_prophet.fit(df_prophet_train)
        forecast = modelo_prophet.predict(df_prophet_future)

        previsoes["Prophet"] = forecast["yhat"].values
        modelos["Prophet"] = modelo_prophet

        # Guardar intervalos também para uso visual.
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
            PREV_DIR / f"{nome}_prophet_intervalos.csv",
            index=False,
        )

        tempo = time.perf_counter() - inicio
        resumo_tempos.append({"serie": nome, "modelo": "Prophet", "tempo_segundos": tempo})
        print(f"[OK] Prophet treinado em {tempo:.2f}s.")
    except Exception as erro:
        print("[INFO] Prophet não foi executado. Motivo:", erro)

    # ---------------------------------------------------------------------
    # Guardar modelos e previsões.
    # ---------------------------------------------------------------------
    joblib.dump(modelos, MODELOS_DIR / f"{nome}_modelos.pkl")
    previsoes.to_csv(PREV_DIR / f"{nome}_previsoes.csv", index_label="data_hora")

    print(f"[OK] Modelos guardados em {MODELOS_DIR / f'{nome}_modelos.pkl'}")
    print(f"[OK] Previsões guardadas em {PREV_DIR / f'{nome}_previsoes.csv'}")

pd.DataFrame(resumo_tempos).to_csv(PREV_DIR / "tempos_treino.csv", index=False)

print("\n" + "=" * 90)
print("[OK] Treino concluído.")
print("=" * 90)
