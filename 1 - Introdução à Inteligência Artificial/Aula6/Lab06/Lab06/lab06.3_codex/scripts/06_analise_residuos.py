# Lab 06.3 — Séries Temporais
# Script 06 — Análise de Resíduos
# Objetivo: analisar erros do melhor modelo por série.
#
# Num bom modelo de previsão, os resíduos devem estar centrados em zero,
# sem padrões temporais claros e com baixa autocorrelação.

from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf


BASE_DIR = Path(__file__).resolve().parents[1]
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
PREV_DIR = ARTEFACTOS_DIR / "previsoes"
TAB_DIR = ARTEFACTOS_DIR / "tabelas"
IMG_DIR = ARTEFACTOS_DIR / "imagens"
REL_DIR = ARTEFACTOS_DIR / "relatorios"

IMG_DIR.mkdir(parents=True, exist_ok=True)
REL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SCRIPT 06 — ANÁLISE DE RESÍDUOS")
print("=" * 90)

with open(TAB_DIR / "melhores_modelos.json", "r", encoding="utf-8") as f:
    melhores = json.load(f)

notas = ["# Notas da Análise de Resíduos\n"]

for nome in ["voltagem", "missoes"]:
    previsoes = pd.read_csv(PREV_DIR / f"{nome}_previsoes.csv", parse_dates=["data_hora"])
    previsoes = previsoes.set_index("data_hora")

    melhor_modelo = melhores[nome]["modelo"]
    residuos = previsoes["real"] - previsoes[melhor_modelo]
    residuos = residuos.dropna()

    print(f"[INFO] {nome}: análise de resíduos do modelo {melhor_modelo}")
    print(residuos.describe())

    # Histograma dos resíduos.
    plt.figure(figsize=(10, 5))
    plt.hist(residuos.values, bins=40)
    plt.axvline(0, linestyle="--", linewidth=1.5)
    plt.title(f"Distribuição dos resíduos — {nome} — {melhor_modelo}")
    plt.xlabel("Resíduo (real - previsto)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"06_{nome}_residuos_histograma.png", dpi=180)
    plt.savefig(IMG_DIR / f"06_{nome}_residuos_histograma.pdf")
    plt.close()

    # Resíduos ao longo do tempo.
    plt.figure(figsize=(14, 5))
    plt.plot(residuos.index, residuos.values, linewidth=1.2)
    plt.axhline(0, linestyle="--", linewidth=1.2)
    plt.title(f"Resíduos ao longo do tempo — {nome} — {melhor_modelo}")
    plt.xlabel("Tempo")
    plt.ylabel("Resíduo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"06_{nome}_residuos_tempo.png", dpi=180)
    plt.savefig(IMG_DIR / f"06_{nome}_residuos_tempo.pdf")
    plt.close()

    # ACF dos resíduos.
    plt.figure(figsize=(12, 5))
    lags = 120 if nome == "voltagem" else 60
    lags = min(lags, max(10, len(residuos) // 3))
    plot_acf(residuos, lags=lags)
    plt.title(f"ACF dos resíduos — {nome} — {melhor_modelo}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"06_{nome}_residuos_acf.png", dpi=180)
    plt.savefig(IMG_DIR / f"06_{nome}_residuos_acf.pdf")
    plt.close()

    media = residuos.mean()
    desvio = residuos.std()
    autocorr_1 = residuos.autocorr(lag=1)

    notas.append(f"\n## {nome}\n")
    notas.append(f"- Melhor modelo: `{melhor_modelo}`.\n")
    notas.append(f"- Média dos resíduos: {media:.4f}.\n")
    notas.append(f"- Desvio-padrão dos resíduos: {desvio:.4f}.\n")
    notas.append(f"- Autocorrelação lag 1 dos resíduos: {autocorr_1:.4f}.\n")
    notas.append("- Interpretação: resíduos com média perto de zero indicam baixo viés global.\n")
    notas.append("- Se a ACF dos resíduos tiver valores relevantes, o modelo ainda deixou estrutura temporal por capturar.\n")

(REL_DIR / "NOTAS_RESIDUOS.md").write_text("".join(notas), encoding="utf-8")

print("\n" + "=" * 90)
print("[OK] Análise de resíduos concluída.")
print("=" * 90)
