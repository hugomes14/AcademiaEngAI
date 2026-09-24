# Lab 06.3 — Séries Temporais
# Script 05 — Gráfico Previsão vs. Real
# Objetivo: representar o histórico, o conjunto de teste e a previsão do melhor modelo.

from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib


BASE_DIR = Path(__file__).resolve().parents[1]
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
OBJ_DIR = ARTEFACTOS_DIR / "objetos"
PREV_DIR = ARTEFACTOS_DIR / "previsoes"
TAB_DIR = ARTEFACTOS_DIR / "tabelas"
IMG_DIR = ARTEFACTOS_DIR / "imagens"

IMG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SCRIPT 05 — GRÁFICO PREVISÃO VS. REAL")
print("=" * 90)

with open(TAB_DIR / "melhores_modelos.json", "r", encoding="utf-8") as f:
    melhores = json.load(f)

notas = []

for nome in ["voltagem", "missoes"]:
    pacote = joblib.load(OBJ_DIR / f"{nome}_dados_processados.pkl")
    y_train = pacote["y_train"]
    y_test = pacote["y_test"]

    previsoes = pd.read_csv(PREV_DIR / f"{nome}_previsoes.csv", parse_dates=["data_hora"])
    previsoes = previsoes.set_index("data_hora")

    melhor_modelo = melhores[nome]["modelo"]
    y_pred = previsoes[melhor_modelo]

    print(f"[INFO] {nome}: melhor modelo = {melhor_modelo}")

    plt.figure(figsize=(15, 6))

    # Para a série minuto a minuto, mostrar apenas cauda do treino para manter o gráfico legível.
    if nome == "voltagem":
        y_train_plot = y_train.tail(24 * 60)
        titulo = "Voltagem — previsão no conjunto de teste"
        eixo_y = "Voltagem (V)"
    else:
        y_train_plot = y_train.tail(120)
        titulo = "Missões diárias — previsão no conjunto de teste"
        eixo_y = "Número de missões"

    plt.plot(y_train_plot.index, y_train_plot.values, label="Histórico de treino", linewidth=1.4)
    plt.plot(y_test.index, y_test.values, label="Real — teste", linewidth=1.8)
    plt.plot(y_test.index, y_pred.values, label=f"Previsão — {melhor_modelo}", linewidth=1.8)

    # Se houver intervalos Prophet disponíveis e o melhor modelo for Prophet, desenhar a banda.
    caminho_intervalos = PREV_DIR / f"{nome}_prophet_intervalos.csv"
    if melhor_modelo == "Prophet" and caminho_intervalos.exists():
        intervalos = pd.read_csv(caminho_intervalos, parse_dates=["ds"])
        plt.fill_between(
            intervalos["ds"],
            intervalos["yhat_lower"],
            intervalos["yhat_upper"],
            alpha=0.2,
            label="Intervalo Prophet 95%",
        )

    plt.title(titulo)
    plt.xlabel("Tempo")
    plt.ylabel(eixo_y)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(IMG_DIR / f"05_{nome}_previsao_vs_real.png", dpi=180)
    plt.savefig(IMG_DIR / f"05_{nome}_previsao_vs_real.pdf")
    plt.close()

    erro_medio = (y_test - y_pred).mean()
    notas.append(f"## {nome}\n")
    notas.append(f"- Melhor modelo por RMSE: `{melhor_modelo}`.\n")
    notas.append(f"- Erro médio (real - previsto): {erro_medio:.4f}.\n")
    if erro_medio > 0:
        notas.append("- Interpretação: em média, o modelo subestima a série no teste.\n")
    elif erro_medio < 0:
        notas.append("- Interpretação: em média, o modelo sobrestima a série no teste.\n")
    else:
        notas.append("- Interpretação: em média, o modelo não apresenta viés global evidente.\n")
    notas.append("\n")

(ARTEFACTOS_DIR / "relatorios" / "NOTAS_PREVISAO.md").write_text("".join(notas), encoding="utf-8")

print("\n" + "=" * 90)
print("[OK] Gráficos de previsão guardados.")
print("=" * 90)
