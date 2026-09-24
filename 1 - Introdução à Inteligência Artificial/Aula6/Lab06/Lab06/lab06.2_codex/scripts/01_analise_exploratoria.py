# -*- coding: utf-8 -*-
"""
Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos
Etapa 01: Análise Exploratória dos Dados (EDA)

Objetivo:
- Ler o dataset de pré-voo.
- Confirmar tipos de dados, valores em falta e desbalanceamento da variável-alvo.
- Criar gráficos simples para apoiar a interpretação inicial.
- Guardar imagens e notas da análise exploratória.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# -------------------------------------------------------------------------
# Configuração de caminhos
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "voos_pre_voo.csv"
EDA_DIR = BASE_DIR / "artefactos" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "incidente_reportado"
NUMERIC_COLS = [
    "idade_aeronave_anos",
    "horas_voo_desde_ultima_manutencao",
    "experiencia_piloto_anos",
]
CATEGORICAL_COLS = ["previsao_turbulencia", "tipo_missao"]

print("=" * 80)
print("ETAPA 01 — ANÁLISE EXPLORATÓRIA")
print("=" * 80)
print(f"Dataset esperado: {DATASET_PATH}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"O dataset não foi encontrado em {DATASET_PATH}. "
        "Coloca o ficheiro voos_pre_voo.csv na pasta data/."
    )

# -------------------------------------------------------------------------
# Leitura e inspeção inicial
# -------------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)

print("\nDimensão do dataset:")
print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")

print("\nPrimeiras linhas:")
print(df.head())

print("\nTipos de dados:")
print(df.dtypes)

print("\nValores em falta por coluna:")
print(df.isna().sum())

print("\nResumo estatístico das variáveis numéricas:")
print(df[NUMERIC_COLS].describe().T)

# -------------------------------------------------------------------------
# Validação simples das colunas esperadas
# -------------------------------------------------------------------------
expected_cols = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]
missing_cols = [col for col in expected_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Faltam colunas obrigatórias no dataset: {missing_cols}")

# -------------------------------------------------------------------------
# Análise do desbalanceamento da variável-alvo
# -------------------------------------------------------------------------
class_counts = df[TARGET].value_counts().sort_index()
class_percent = df[TARGET].value_counts(normalize=True).sort_index() * 100
class_summary = pd.DataFrame({
    "classe": class_counts.index,
    "contagem": class_counts.values,
    "percentagem": class_percent.round(2).values,
})

print("\nDistribuição da variável-alvo:")
print(class_summary.to_string(index=False))

class_summary.to_csv(EDA_DIR / "distribuicao_classes.csv", index=False, encoding="utf-8")
class_summary.to_markdown(EDA_DIR / "distribuicao_classes.md", index=False)

plt.figure(figsize=(7, 5))
ax = plt.gca()
bars = ax.bar(class_summary["classe"].astype(str), class_summary["contagem"])
plt.title("Distribuição da variável-alvo: incidente_reportado")
plt.xlabel("Classe (0 = sem incidente, 1 = incidente)")
plt.ylabel("Número de voos")
ax.bar_label(bars)
plt.tight_layout()
plt.savefig(EDA_DIR / "01_distribuicao_alvo.png", dpi=160)
plt.savefig(EDA_DIR / "01_distribuicao_alvo.pdf")
plt.close()

# -------------------------------------------------------------------------
# Distribuição das variáveis numéricas por classe
# -------------------------------------------------------------------------
for col in NUMERIC_COLS:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=col, hue=TARGET, bins=30, kde=True, element="step")
    plt.title(f"Distribuição de {col} por classe")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(EDA_DIR / f"02_hist_{col}_por_classe.png", dpi=160)
    plt.savefig(EDA_DIR / f"02_hist_{col}_por_classe.pdf")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET, y=col)
    plt.title(f"Boxplot de {col} por classe")
    plt.xlabel("Classe")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(EDA_DIR / f"03_boxplot_{col}_por_classe.png", dpi=160)
    plt.savefig(EDA_DIR / f"03_boxplot_{col}_por_classe.pdf")
    plt.close()

# -------------------------------------------------------------------------
# Relação das variáveis categóricas com a variável-alvo
# -------------------------------------------------------------------------
for col in CATEGORICAL_COLS:
    tabela = pd.crosstab(df[col], df[TARGET], margins=True)
    tabela_percent = pd.crosstab(df[col], df[TARGET], normalize="index") * 100
    print(f"\nTabela de contingência para {col}:")
    print(tabela)
    print(f"\nPercentagens por categoria para {col}:")
    print(tabela_percent.round(2))

    tabela.to_csv(EDA_DIR / f"contingencia_{col}.csv", encoding="utf-8")
    tabela_percent.round(2).to_csv(EDA_DIR / f"contingencia_percent_{col}.csv", encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plot_df = pd.crosstab(df[col], df[TARGET])
    ax = plot_df.plot(kind="bar", ax=plt.gca())
    plt.title(f"{col} por classe de incidente")
    plt.xlabel(col)
    plt.ylabel("Número de voos")
    plt.xticks(rotation=20)
    plt.legend(title=TARGET)
    plt.tight_layout()
    plt.savefig(EDA_DIR / f"04_countplot_{col}_por_classe.png", dpi=160)
    plt.savefig(EDA_DIR / f"04_countplot_{col}_por_classe.pdf")
    plt.close()

# -------------------------------------------------------------------------
# Correlações entre variáveis numéricas e alvo
# -------------------------------------------------------------------------
correlation_cols = NUMERIC_COLS + [TARGET]
correlation_matrix = df[correlation_cols].corr(numeric_only=True)
print("\nMatriz de correlação:")
print(correlation_matrix.round(3))
correlation_matrix.round(4).to_csv(EDA_DIR / "correlacoes_numericas.csv", encoding="utf-8")

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues", square=True)
plt.title("Correlações entre variáveis numéricas e alvo")
plt.tight_layout()
plt.savefig(EDA_DIR / "05_heatmap_correlacoes.png", dpi=160)
plt.savefig(EDA_DIR / "05_heatmap_correlacoes.pdf")
plt.close()

# -------------------------------------------------------------------------
# Notas automáticas da EDA
# -------------------------------------------------------------------------
minority_rate = class_percent.get(1, 0.0)
notes = []
notes.append("# Notas da Análise Exploratória\n")
notes.append(f"- O dataset tem **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.")
notes.append(f"- A variável-alvo é `{TARGET}`.")
notes.append(f"- A classe positiva (`1`, incidente) representa **{minority_rate:.2f}%** dos registos.")
notes.append("- Existe desbalanceamento de classes; por isso, a Accuracy deve ser interpretada com cuidado.")
notes.append("- Em problemas de segurança, falsos negativos são críticos: voos de risco classificados como seguros.")
notes.append("- O pré-processamento deve ajustar codificadores e escalonadores apenas no treino para evitar data leakage.")
notes.append("- O escalonamento é importante para KNN e SVM porque estes modelos dependem de distâncias ou margens.")

(EDA_DIR / "notas_eda.md").write_text("\n".join(notes), encoding="utf-8")

print("\nArtefactos da EDA guardados em:")
print(EDA_DIR)
print("\nEtapa 01 concluída com sucesso.")
