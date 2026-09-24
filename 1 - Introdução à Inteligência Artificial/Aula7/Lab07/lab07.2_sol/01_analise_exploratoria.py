# 01_analise_exploratoria.py
# Lab 07.2 — Redução de Dimensionalidade
# Etapa 1: análise exploratória dos sensores de voo.
#
# Nota pedagógica:
# - Este ficheiro corre de forma direta, sem funções e sem main.
# - O objetivo é produzir tabelas e imagens que ajudem a perceber a escala,
#   a distribuição e a correlação entre sensores antes do PCA.

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dados" / "sensores_voo_alta_dim.csv"
IMAGENS_DIR = BASE_DIR / "imagens"
TABELAS_DIR = BASE_DIR / "tabelas"

IMAGENS_DIR.mkdir(exist_ok=True)
TABELAS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("LAB 07.2 — ETAPA 1: ANÁLISE EXPLORATÓRIA")
print("=" * 80)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset carregado: {DATASET_PATH}")
print(f"Dimensão do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
print("\nPrimeiras linhas:")
print(df.head())

target_col = "tipo_manobra" if "tipo_manobra" in df.columns else None
numeric_cols = df.select_dtypes(include="number").columns.tolist()

print("\nColunas numéricas detetadas:")
print(numeric_cols)

if target_col:
    print(f"\nColuna de rótulo opcional detetada: {target_col}")
    print("Esta coluna será usada para colorir visualizações e para LDA, não para treinar PCA.")
    print("\nDistribuição da coluna tipo_manobra:")
    print(df[target_col].value_counts(dropna=False))
else:
    print("\nNão foi encontrada coluna tipo_manobra. As visualizações seguem sem cor por classe.")

print("\nValores em falta por coluna:")
print(df.isna().sum())

print("\nDuplicados no dataset:")
print(df.duplicated().sum())

resumo = df[numeric_cols].describe().T
resumo["skewness"] = df[numeric_cols].skew()

# IQR e contagem de outliers por feature.
q1 = df[numeric_cols].quantile(0.25)
q3 = df[numeric_cols].quantile(0.75)
iqr = q3 - q1
limite_inf = q1 - 1.5 * iqr
limite_sup = q3 + 1.5 * iqr

outliers = ((df[numeric_cols] < limite_inf) | (df[numeric_cols] > limite_sup)).sum()
resumo["outliers_iqr"] = outliers
resumo["percentagem_outliers_iqr"] = (outliers / len(df) * 100).round(2)

resumo.to_csv(TABELAS_DIR / "01_resumo_estatistico_sensores.csv", encoding="utf-8")
resumo.to_markdown(TABELAS_DIR / "01_resumo_estatistico_sensores.md")

print("\nResumo estatístico guardado em tabelas/01_resumo_estatistico_sensores.csv")
print(resumo.head(10))

# Histogramas de todas as features numéricas.
n_cols_plot = 5
n_rows_plot = (len(numeric_cols) + n_cols_plot - 1) // n_cols_plot
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(20, max(10, n_rows_plot * 2.6)))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=30)
    axes[i].set_title(col, fontsize=8)
    axes[i].tick_params(axis="both", labelsize=7)

for j in range(len(numeric_cols), len(axes)):
    axes[j].axis("off")

fig.suptitle("Distribuição das features numéricas", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(IMAGENS_DIR / "01_histogramas_sensores.png", dpi=180)
fig.savefig(IMAGENS_DIR / "01_histogramas_sensores.pdf")
plt.close(fig)

print("Imagem guardada: imagens/01_histogramas_sensores.png")

# Boxplot agregado. Ajuda a perceber escalas e outliers.
fig, ax = plt.subplots(figsize=(18, 8))
df[numeric_cols].plot(kind="box", ax=ax, rot=90)
ax.set_title("Boxplots das features numéricas — observar escala e outliers")
ax.set_ylabel("Valor original")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "01_boxplots_sensores.png", dpi=180)
fig.savefig(IMAGENS_DIR / "01_boxplots_sensores.pdf")
plt.close(fig)

print("Imagem guardada: imagens/01_boxplots_sensores.png")

# Heatmap de correlações.
corr = df[numeric_cols].corr()
corr.to_csv(TABELAS_DIR / "01_matriz_correlacao.csv", encoding="utf-8")

fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de correlação entre sensores")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "01_heatmap_correlacoes.png", dpi=180)
fig.savefig(IMAGENS_DIR / "01_heatmap_correlacoes.pdf")
plt.close(fig)

print("Imagem guardada: imagens/01_heatmap_correlacoes.png")

# Pairplot com um subconjunto de sensores de maior variância para evitar uma imagem ilegível.
variancias = df[numeric_cols].var().sort_values(ascending=False)
cols_pairplot = variancias.head(6).index.tolist()
print("\nSensores selecionados para o pairplot, por maior variância:")
print(cols_pairplot)

pairplot_df = df[cols_pairplot + ([target_col] if target_col else [])].copy()

if target_col:
    g = sns.pairplot(pairplot_df, hue=target_col, diag_kind="hist", corner=True, plot_kws={"alpha": 0.7})
else:
    g = sns.pairplot(pairplot_df, diag_kind="hist", corner=True, plot_kws={"alpha": 0.7})

g.fig.suptitle("Pairplot de sensores com maior variância", y=1.02)
g.fig.savefig(IMAGENS_DIR / "01_pairplot_sensores_maior_variancia.png", dpi=160, bbox_inches="tight")
g.fig.savefig(IMAGENS_DIR / "01_pairplot_sensores_maior_variancia.pdf", bbox_inches="tight")
plt.close(g.fig)

print("Imagem guardada: imagens/01_pairplot_sensores_maior_variancia.png")

notas = []
notas.append("# Notas da EDA — Lab 07.2\n")
notas.append(f"- Dataset com {df.shape[0]} linhas e {df.shape[1]} colunas.")
notas.append(f"- Foram detetadas {len(numeric_cols)} features numéricas.")
if target_col:
    notas.append(f"- A coluna `{target_col}` existe e será usada apenas para visualização e LDA.")
notas.append("- O PCA é sensível à escala, porque procura direções de maior variância.")
notas.append("- Sem escalonamento, sensores com valores absolutos maiores podem dominar os componentes principais.")
notas.append("- Outliers podem distorcer a variância e, por isso, alterar os componentes principais.")
notas.append("- Correlações fortes entre sensores indicam redundância potencial, que o PCA pode comprimir.")

(TABELAS_DIR / "01_notas_eda.md").write_text("\n".join(notas), encoding="utf-8")

print("\nNotas guardadas em tabelas/01_notas_eda.md")
print("\nEtapa 1 concluída com sucesso.")
