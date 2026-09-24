# Lab 06.3 — Séries Temporais
# Script 01 — Análise Exploratória (EDA)
# Objetivo: conhecer as duas séries temporais, verificar frequência, lacunas,
# duplicados, estacionaridade, sazonalidade e autocorrelação.

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


# -------------------------------------------------------------------------
# Configuração de caminhos
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DADOS_DIR = BASE_DIR / "dados"
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
IMG_DIR = ARTEFACTOS_DIR / "imagens"
REL_DIR = ARTEFACTOS_DIR / "relatorios"
OBJ_DIR = ARTEFACTOS_DIR / "objetos"

IMG_DIR.mkdir(parents=True, exist_ok=True)
REL_DIR.mkdir(parents=True, exist_ok=True)
OBJ_DIR.mkdir(parents=True, exist_ok=True)

ficheiro_voltagem = DADOS_DIR / "voltagem_bateria.csv"
ficheiro_missoes = DADOS_DIR / "missoes_diarias.csv"

print("=" * 90)
print("SCRIPT 01 — ANÁLISE EXPLORATÓRIA DE SÉRIES TEMPORAIS")
print("=" * 90)

# -------------------------------------------------------------------------
# Leitura dos dados
# -------------------------------------------------------------------------
voltagem = pd.read_csv(ficheiro_voltagem)
missoes = pd.read_csv(ficheiro_missoes)

print("\n[INFO] Ficheiros carregados com sucesso.")
print(f"[INFO] voltagem_bateria.csv: {voltagem.shape[0]} linhas, {voltagem.shape[1]} colunas")
print(f"[INFO] missoes_diarias.csv: {missoes.shape[0]} linhas, {missoes.shape[1]} colunas")

# -------------------------------------------------------------------------
# Conversão das colunas temporais
# -------------------------------------------------------------------------
voltagem["timestamp"] = pd.to_datetime(voltagem["timestamp"], errors="coerce")
missoes["data"] = pd.to_datetime(missoes["data"], errors="coerce")

voltagem = voltagem.dropna(subset=["timestamp", "voltagem"]).sort_values("timestamp")
missoes = missoes.dropna(subset=["data", "num_missoes"]).sort_values("data")

voltagem = voltagem.set_index("timestamp")
missoes = missoes.set_index("data")

# Garantir ordem cronológica e remover duplicados, se existirem.
voltagem = voltagem[~voltagem.index.duplicated(keep="first")]
missoes = missoes[~missoes.index.duplicated(keep="first")]

series = {
    "voltagem": {
        "df": voltagem,
        "target": "voltagem",
        "freq_esperada": "min",
        "periodo_decomposicao": 1440,
        "periodo_decomposicao_alternativo": 24,
        "lags_acf": 120,
        "descricao": "Voltagem da bateria minuto a minuto",
    },
    "missoes": {
        "df": missoes,
        "target": "num_missoes",
        "freq_esperada": "D",
        "periodo_decomposicao": 7,
        "periodo_decomposicao_alternativo": 7,
        "lags_acf": 60,
        "descricao": "Número de missões diárias",
    },
}

linhas_relatorio = []
linhas_relatorio.append("# Notas da Análise Exploratória\n")
linhas_relatorio.append("Este ficheiro foi gerado por `01_analise_exploratoria.py`.\n")

for nome, cfg in series.items():
    df = cfg["df"].copy()
    target = cfg["target"]

    print("\n" + "-" * 90)
    print(f"SÉRIE: {nome.upper()} — {cfg['descricao']}")
    print("-" * 90)

    print("\n[AMOSTRA]")
    print(df.head())
    print("\n[ESTATÍSTICAS]")
    print(df[target].describe())

    # Frequência inferida e verificação de lacunas.
    freq_inferida = pd.infer_freq(df.index)
    print(f"\n[INFO] Frequência inferida pelo pandas: {freq_inferida}")

    if nome == "voltagem":
        indice_completo = pd.date_range(df.index.min(), df.index.max(), freq="min")
    else:
        indice_completo = pd.date_range(df.index.min(), df.index.max(), freq="D")

    lacunas = indice_completo.difference(df.index)
    print(f"[INFO] Duplicados no índice temporal: {df.index.duplicated().sum()}")
    print(f"[INFO] Lacunas temporais detetadas: {len(lacunas)}")

    # Guardar notas.
    linhas_relatorio.append(f"\n## Série: {nome}\n")
    linhas_relatorio.append(f"- Descrição: {cfg['descricao']}\n")
    linhas_relatorio.append(f"- Linhas: {len(df)}\n")
    linhas_relatorio.append(f"- Período: {df.index.min()} a {df.index.max()}\n")
    linhas_relatorio.append(f"- Frequência inferida: {freq_inferida}\n")
    linhas_relatorio.append(f"- Lacunas temporais: {len(lacunas)}\n")
    linhas_relatorio.append(f"- Média: {df[target].mean():.4f}\n")
    linhas_relatorio.append(f"- Mediana: {df[target].median():.4f}\n")
    linhas_relatorio.append(f"- Desvio-padrão: {df[target].std():.4f}\n")

    # ---------------------------------------------------------------------
    # Gráfico da série temporal.
    # Para a série minuto a minuto, cria-se também uma versão suavizada por hora
    # para melhorar a leitura visual.
    # ---------------------------------------------------------------------
    plt.figure(figsize=(14, 6))
    if nome == "voltagem":
        df_plot = df[target].resample("h").mean()
        plt.plot(df_plot.index, df_plot.values, linewidth=1.2)
        plt.title("Voltagem da bateria — média horária")
        plt.ylabel("Voltagem (V)")
    else:
        plt.plot(df.index, df[target], linewidth=1.4)
        plt.title("Número de missões diárias")
        plt.ylabel("Número de missões")

    plt.xlabel("Tempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"01_{nome}_serie_temporal.png", dpi=180)
    plt.savefig(IMG_DIR / f"01_{nome}_serie_temporal.pdf")
    plt.close()

    # ---------------------------------------------------------------------
    # Histograma e boxplot da variável alvo.
    # ---------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.hist(df[target].dropna().values, bins=40)
    plt.title(f"Distribuição — {target}")
    plt.xlabel(target)
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"01_{nome}_distribuicao.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot(df[target].dropna().values, vert=False)
    plt.title(f"Boxplot — {target}")
    plt.xlabel(target)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"01_{nome}_boxplot.png", dpi=180)
    plt.close()

    # ---------------------------------------------------------------------
    # Teste de estacionaridade ADF.
    # H0: a série tem raiz unitária, logo não é estacionária.
    # p-value baixo (< 0.05) sugere estacionaridade.
    # ---------------------------------------------------------------------
    serie_adf = df[target].dropna()
    if len(serie_adf) > 5000:
        serie_adf = serie_adf.iloc[::5]

    try:
        adf_resultado = adfuller(serie_adf, autolag="AIC")
        print("\n[ADF — TESTE DE ESTACIONARIDADE]")
        print(f"Estatística ADF: {adf_resultado[0]:.4f}")
        print(f"p-value: {adf_resultado[1]:.6f}")
        print("Interpretação: p-value < 0.05 sugere série estacionária.")
        linhas_relatorio.append(f"- ADF statistic: {adf_resultado[0]:.4f}\n")
        linhas_relatorio.append(f"- ADF p-value: {adf_resultado[1]:.6f}\n")
    except Exception as erro:
        print(f"[AVISO] Não foi possível calcular o teste ADF: {erro}")
        linhas_relatorio.append(f"- ADF: não calculado ({erro})\n")

    # ---------------------------------------------------------------------
    # Decomposição.
    # Para voltagem, usa-se uma amostra horária para obter um gráfico mais leve.
    # ---------------------------------------------------------------------
    try:
        if nome == "voltagem":
            serie_decomp = df[target].resample("h").mean().dropna()
            periodo = cfg["periodo_decomposicao_alternativo"]
        else:
            serie_decomp = df[target].dropna()
            periodo = cfg["periodo_decomposicao"]

        if len(serie_decomp) >= 2 * periodo:
            decomposicao = seasonal_decompose(serie_decomp, model="additive", period=periodo)
            fig = decomposicao.plot()
            fig.set_size_inches(14, 9)
            fig.suptitle(f"Decomposição da série — {nome}", y=1.02)
            plt.tight_layout()
            plt.savefig(IMG_DIR / f"01_{nome}_decomposicao.png", dpi=180)
            plt.savefig(IMG_DIR / f"01_{nome}_decomposicao.pdf")
            plt.close()
            print(f"[INFO] Decomposição guardada com período={periodo}.")
        else:
            print("[AVISO] Série demasiado curta para decomposição.")
    except Exception as erro:
        print(f"[AVISO] Não foi possível gerar decomposição: {erro}")

    # ---------------------------------------------------------------------
    # ACF e PACF.
    # ACF ajuda a perceber autocorrelação.
    # PACF ajuda a escolher lags relevantes em modelos autorregressivos.
    # ---------------------------------------------------------------------
    try:
        serie_acf = df[target].dropna()
        if len(serie_acf) > 5000:
            serie_acf = serie_acf.iloc[::5]

        max_lags = min(cfg["lags_acf"], max(10, len(serie_acf) // 3))

        plt.figure(figsize=(12, 5))
        plot_acf(serie_acf, lags=max_lags)
        plt.title(f"ACF — {nome}")
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"01_{nome}_acf.png", dpi=180)
        plt.close()

        plt.figure(figsize=(12, 5))
        plot_pacf(serie_acf, lags=max_lags, method="ywm")
        plt.title(f"PACF — {nome}")
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"01_{nome}_pacf.png", dpi=180)
        plt.close()

        print(f"[INFO] ACF/PACF guardados com {max_lags} lags.")
    except Exception as erro:
        print(f"[AVISO] Não foi possível gerar ACF/PACF: {erro}")

# Guardar dados limpos básicos para reutilização.
voltagem.to_csv(ARTEFACTOS_DIR / "objetos" / "voltagem_limpa.csv")
missoes.to_csv(ARTEFACTOS_DIR / "objetos" / "missoes_limpas.csv")

(REL_DIR / "NOTAS_EDA.md").write_text("".join(linhas_relatorio), encoding="utf-8")

print("\n" + "=" * 90)
print("[OK] EDA concluída. Imagens e notas guardadas em artefactos/.")
print("=" * 90)
