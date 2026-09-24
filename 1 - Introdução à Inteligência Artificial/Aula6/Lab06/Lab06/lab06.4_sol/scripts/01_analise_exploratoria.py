# 01_analise_exploratoria.py
# Lab 06.4 — Modelos Ensemble
# Etapa: Análise Exploratória dos Dados
#
# Este script lê o dataset original, valida as colunas esperadas, analisa o
# desbalanceamento da variável-alvo e gera gráficos para apoiar as decisões
# de modelação.
#
# Nota pedagógica:
# - A variável-alvo é binária: 0 = sem incidente, 1 = incidente.
# - Como o dataset é desbalanceado, a Accuracy pode ser enganadora.
# - Métricas como Recall, F1 e ROC-AUC são mais úteis para avaliar risco.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dados" / "voos_pre_voo.csv"
EDA_DIR = BASE_DIR / "artefactos" / "eda"
FIG_DIR = BASE_DIR / "artefactos" / "figuras"

EDA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.4 — ETAPA 1: ANÁLISE EXPLORATÓRIA")
print("=" * 80)

print(f"\nA ler dataset em: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

print("\nDimensão do dataset:")
print(df.shape)

print("\nPrimeiras 5 linhas:")
print(df.head())

print("\nTipos de dados:")
print(df.dtypes)

colunas_esperadas = [
    "idade_aeronave_anos",
    "horas_voo_desde_ultima_manutencao",
    "previsao_turbulencia",
    "tipo_missao",
    "experiencia_piloto_anos",
    "incidente_reportado",
]

colunas_em_falta = [col for col in colunas_esperadas if col not in df.columns]
if colunas_em_falta:
    raise ValueError(f"Colunas em falta no dataset: {colunas_em_falta}")

target = "incidente_reportado"
numericas = [
    "idade_aeronave_anos",
    "horas_voo_desde_ultima_manutencao",
    "experiencia_piloto_anos",
]
categoricas = ["previsao_turbulencia", "tipo_missao"]

print("\nValores em falta por coluna:")
print(df.isna().sum())

print("\nDuplicados:")
print(df.duplicated().sum())

print("\nDistribuição da variável-alvo:")
contagens = df[target].value_counts().sort_index()
percentagens = df[target].value_counts(normalize=True).sort_index() * 100
distribuicao_alvo = pd.DataFrame({
    "classe": contagens.index,
    "contagem": contagens.values,
    "percentagem": percentagens.values,
})
print(distribuicao_alvo)

distribuicao_alvo.to_csv(EDA_DIR / "distribuicao_alvo.csv", index=False)
distribuicao_alvo.to_markdown(EDA_DIR / "distribuicao_alvo.md", index=False)

plt.figure(figsize=(8, 5))
plt.bar(distribuicao_alvo["classe"].astype(str), distribuicao_alvo["contagem"])
plt.title("Distribuição da variável-alvo: incidente_reportado")
plt.xlabel("Classe")
plt.ylabel("Número de voos")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_distribuicao_alvo.png", dpi=200)
plt.savefig(FIG_DIR / "eda_distribuicao_alvo.pdf")
plt.close()

print("\nResumo estatístico das variáveis numéricas:")
resumo_numerico = df[numericas].describe().T
print(resumo_numerico)
resumo_numerico.to_csv(EDA_DIR / "resumo_variaveis_numericas.csv")
resumo_numerico.to_markdown(EDA_DIR / "resumo_variaveis_numericas.md")

plt.figure(figsize=(10, 8))
for i, coluna in enumerate(numericas, start=1):
    plt.subplot(len(numericas), 1, i)
    sns.histplot(data=df, x=coluna, hue=target, kde=True, bins=30)
    plt.title(f"Distribuição de {coluna} por classe")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_distribuicoes_numericas_por_classe.png", dpi=200)
plt.savefig(FIG_DIR / "eda_distribuicoes_numericas_por_classe.pdf")
plt.close()

plt.figure(figsize=(10, 8))
for i, coluna in enumerate(numericas, start=1):
    plt.subplot(len(numericas), 1, i)
    sns.boxplot(data=df, x=target, y=coluna)
    plt.title(f"{coluna} por classe do incidente")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_boxplots_numericas_por_classe.png", dpi=200)
plt.savefig(FIG_DIR / "eda_boxplots_numericas_por_classe.pdf")
plt.close()

for coluna in categoricas:
    tabela = pd.crosstab(df[coluna], df[target], normalize="index") * 100
    print(f"\nPercentagem de classes por categoria em {coluna}:")
    print(tabela.round(2))
    tabela.to_csv(EDA_DIR / f"distribuicao_{coluna}_por_classe.csv")
    tabela.to_markdown(EDA_DIR / f"distribuicao_{coluna}_por_classe.md")

    plt.figure(figsize=(8, 5))
    tabela.plot(kind="bar", stacked=True)
    plt.title(f"Distribuição percentual de {target} por {coluna}")
    plt.xlabel(coluna)
    plt.ylabel("Percentagem")
    plt.legend(title=target)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"eda_{coluna}_por_classe.png", dpi=200)
    plt.savefig(FIG_DIR / f"eda_{coluna}_por_classe.pdf")
    plt.close()

correlacoes = df[numericas + [target]].corr(numeric_only=True)
print("\nCorrelação entre variáveis numéricas e a variável-alvo:")
print(correlacoes)

correlacoes.to_csv(EDA_DIR / "correlacoes.csv")
correlacoes.to_markdown(EDA_DIR / "correlacoes.md")

plt.figure(figsize=(8, 6))
sns.heatmap(correlacoes, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre variáveis numéricas e incidente_reportado")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_heatmap_correlacoes.png", dpi=200)
plt.savefig(FIG_DIR / "eda_heatmap_correlacoes.pdf")
plt.close()

notas = f"""# Notas da Análise Exploratória

## Dataset

- Linhas: {df.shape[0]}
- Colunas: {df.shape[1]}
- Variável-alvo: `{target}`

## Desbalanceamento

{distribuicao_alvo.to_markdown(index=False)}

Este dataset é desbalanceado, porque a classe de incidentes representa uma minoria.
Por isso, a Accuracy não deve ser usada isoladamente. Um modelo que prevê sempre
"sem incidente" pode ter uma Accuracy elevada, mas não tem utilidade operacional.

## Decisões para as etapas seguintes

- Usar `train_test_split` estratificado.
- Aplicar codificação ordinal à turbulência, porque existe ordem natural: Baixa < Média < Alta.
- Aplicar one-hot encoding ao tipo de missão, porque não existe ordem natural.
- Usar `class_weight` nos modelos que suportam este parâmetro.
- Avaliar com F1, Recall, ROC-AUC e PR-AUC.
"""

(EDA_DIR / "notas_eda.md").write_text(notas, encoding="utf-8")

print("\nArtefactos EDA guardados em:")
print(EDA_DIR)
print(FIG_DIR)
print("\nEtapa 1 concluída com sucesso.")
