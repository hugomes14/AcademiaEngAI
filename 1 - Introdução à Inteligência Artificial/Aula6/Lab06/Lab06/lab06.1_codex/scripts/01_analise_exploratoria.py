# 01_analise_exploratoria.py
# Lab 06.1 - Regressão — Estimar a Duração de Voo
# Objetivo: explorar o dataset, validar colunas, analisar a variável-alvo e criar gráficos de apoio.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Configuração de caminhos
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "voos_telemetria.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figuras"
TAB_DIR = OUTPUT_DIR / "tabelas"
NOTE_DIR = OUTPUT_DIR / "notas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAME = "duracao_voo_min"
NUMERIC_COLUMNS = ["distancia_planeada", "carga_util_kg", "altitude_media_m"]
CATEGORICAL_COLUMNS = ["condicao_meteo"]
EXPECTED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_NAME]

print("=" * 80)
print("01 - ANÁLISE EXPLORATÓRIA DE DADOS")
print("=" * 80)
print(f"Dataset esperado: {DATASET_PATH}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset não encontrado em {DATASET_PATH}. Coloca o ficheiro voos_telemetria.csv na pasta data/."
    )

# -----------------------------
# Carregamento e validação básica
# -----------------------------
df = pd.read_csv(DATASET_PATH)
print(f"\nDimensão do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
print("\nPrimeiras 5 linhas:")
print(df.head())
print("\nTipos de dados:")
print(df.dtypes)

missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
if missing_columns:
    raise ValueError(f"Colunas em falta no dataset: {missing_columns}")

missing_values = df.isna().sum().sort_values(ascending=False)
print("\nValores em falta por coluna:")
print(missing_values)
missing_values.to_csv(TAB_DIR / "01_valores_em_falta.csv", header=["valores_em_falta"])

# -----------------------------
# Estatísticas descritivas
# -----------------------------
descriptive_stats = df[NUMERIC_COLUMNS + [TARGET_NAME]].describe().T
print("\nEstatísticas descritivas das variáveis numéricas:")
print(descriptive_stats)
descriptive_stats.to_csv(TAB_DIR / "01_estatisticas_descritivas.csv")

categorical_summary = df[CATEGORICAL_COLUMNS[0]].value_counts(dropna=False).rename_axis("categoria").reset_index(name="contagem")
print("\nResumo da variável categórica condicao_meteo:")
print(categorical_summary)
categorical_summary.to_csv(TAB_DIR / "01_resumo_condicao_meteo.csv", index=False)

# -----------------------------
# Distribuição da variável-alvo
# -----------------------------
target_skewness = df[TARGET_NAME].skew()
q1 = df[TARGET_NAME].quantile(0.25)
q3 = df[TARGET_NAME].quantile(0.75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
outliers_target = df[(df[TARGET_NAME] < lower_limit) | (df[TARGET_NAME] > upper_limit)]

print("\nAnálise da variável-alvo:")
print(f"Skewness de {TARGET_NAME}: {target_skewness:.4f}")
print(f"Limite inferior IQR: {lower_limit:.4f}")
print(f"Limite superior IQR: {upper_limit:.4f}")
print(f"Número de potenciais outliers no alvo: {len(outliers_target)}")

pd.DataFrame({
    "metrica": ["skewness", "q1", "q3", "iqr", "limite_inferior", "limite_superior", "n_outliers"],
    "valor": [target_skewness, q1, q3, iqr, lower_limit, upper_limit, len(outliers_target)]
}).to_csv(TAB_DIR / "01_alvo_skewness_outliers.csv", index=False)

plt.figure(figsize=(10, 6))
plt.hist(df[TARGET_NAME], bins=30, edgecolor="black", alpha=0.8)
plt.title("Distribuição da duração do voo")
plt.xlabel("Duração do voo (minutos)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_distribuicao_duracao_voo.png", dpi=300)
plt.savefig(FIG_DIR / "01_distribuicao_duracao_voo.pdf")
plt.close()

plt.figure(figsize=(10, 4))
plt.boxplot(df[TARGET_NAME], vert=False)
plt.title("Boxplot da duração do voo")
plt.xlabel("Duração do voo (minutos)")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_boxplot_duracao_voo.png", dpi=300)
plt.savefig(FIG_DIR / "01_boxplot_duracao_voo.pdf")
plt.close()

# -----------------------------
# Relação entre numéricas e alvo
# -----------------------------
for column in NUMERIC_COLUMNS:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[column], df[TARGET_NAME], alpha=0.75)
    plt.title(f"{column} vs. {TARGET_NAME}")
    plt.xlabel(column)
    plt.ylabel("Duração do voo (minutos)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"01_dispersao_{column}_vs_duracao.png", dpi=300)
    plt.savefig(FIG_DIR / f"01_dispersao_{column}_vs_duracao.pdf")
    plt.close()

# -----------------------------
# Relação entre categórica e alvo
# -----------------------------
plt.figure(figsize=(8, 6))
ordered_categories = ["Bom", "Moderado", "Adverso"]
boxplot_data = [df.loc[df["condicao_meteo"] == category, TARGET_NAME] for category in ordered_categories]
plt.boxplot(boxplot_data, tick_labels=ordered_categories)
plt.title("Duração do voo por condição meteorológica")
plt.xlabel("Condição meteorológica")
plt.ylabel("Duração do voo (minutos)")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_boxplot_condicao_meteo_vs_duracao.png", dpi=300)
plt.savefig(FIG_DIR / "01_boxplot_condicao_meteo_vs_duracao.pdf")
plt.close()

# -----------------------------
# Correlações numéricas
# -----------------------------
correlation_matrix = df[NUMERIC_COLUMNS + [TARGET_NAME]].corr(numeric_only=True)
print("\nMatriz de correlação:")
print(correlation_matrix)
correlation_matrix.to_csv(TAB_DIR / "01_matriz_correlacao.csv")

plt.figure(figsize=(8, 6))
im = plt.imshow(correlation_matrix.values)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.3f}", ha="center", va="center")
plt.title("Correlação entre variáveis numéricas")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_heatmap_correlacao.png", dpi=300)
plt.savefig(FIG_DIR / "01_heatmap_correlacao.pdf")
plt.close()

# -----------------------------
# Notas interpretativas automáticas
# -----------------------------
notes = []
notes.append("# Notas da Análise Exploratória\n")
notes.append(f"- O dataset tem **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.\n")
notes.append(f"- A variável-alvo é `{TARGET_NAME}`, medida em minutos.\n")
notes.append(f"- A skewness da variável-alvo é **{target_skewness:.4f}**. Valores afastados de zero sugerem assimetria.\n")
notes.append(f"- Foram detetados **{len(outliers_target)} potenciais outliers** na duração do voo pelo critério IQR.\n")
notes.append("- O RMSE tende a penalizar mais erros grandes do que o MAE, por isso é sensível a outliers.\n")
notes.append("- O data leakage deve ser evitado: qualquer transformação aprendida nos dados deve ser ajustada apenas no treino e depois aplicada ao teste.\n")
notes.append("- A condição meteorológica tem uma ordem natural: Bom < Moderado < Adverso. Neste projeto será codificada de forma ordinal.\n")

(NOTE_DIR / "01_notas_eda.md").write_text("".join(notes), encoding="utf-8")

print("\nFicheiros gerados:")
print(f"- Figuras: {FIG_DIR}")
print(f"- Tabelas: {TAB_DIR}")
print(f"- Notas: {NOTE_DIR / '01_notas_eda.md'}")
print("\n01 - Análise exploratória concluída com sucesso.")
