# 01_analise_exploratoria.py
# Lab 07.1 — Clustering: Descoberta de Perfis Operacionais de Voo
# Etapa 1: Análise Exploratória dos Dados (EDA)
#
# Objetivo pedagógico:
# - Verificar a estrutura do dataset.
# - Analisar distribuição, escala, assimetria e outliers das variáveis.
# - Criar gráficos úteis antes do clustering.
# - Reforçar por que o escalonamento é essencial em algoritmos baseados em distância.

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
DATASET_PATH = BASE_DIR / "data" / "voos_telemetria_completa.csv"

FIG_DIR = BASE_DIR / "outputs" / "figures" / "eda"
TABLE_DIR = BASE_DIR / "outputs" / "tables"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 07.1 — ETAPA 1: ANÁLISE EXPLORATÓRIA")
print("=" * 80)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("\n[INFO] Dataset carregado com sucesso.")
print(f"[INFO] Caminho: {DATASET_PATH}")
print(f"[INFO] Dimensão: {df.shape[0]} linhas x {df.shape[1]} colunas")

print("\n[INFO] Primeiras linhas:")
print(df.head())

print("\n[INFO] Tipos de dados:")
print(df.dtypes)

print("\n[INFO] Valores em falta por coluna:")
print(df.isna().sum())

print("\n[INFO] Duplicados:")
print(f"Número de linhas duplicadas: {df.duplicated().sum()}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\n[INFO] Colunas numéricas detetadas:")
print(numeric_cols)

print("\n[INFO] Colunas categóricas detetadas:")
print(categorical_cols if categorical_cols else "Nenhuma coluna categórica detetada.")

if len(numeric_cols) < 2:
    raise ValueError("O clustering precisa de pelo menos duas variáveis numéricas úteis.")

desc = df[numeric_cols].describe().T
desc["skewness"] = df[numeric_cols].skew(numeric_only=True)

outlier_rows = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    n_outliers = int(((df[col] < lim_inf) | (df[col] > lim_sup)).sum())
    outlier_rows.append({
        "feature": col,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": lim_inf,
        "limite_superior": lim_sup,
        "n_outliers_iqr": n_outliers,
        "percent_outliers_iqr": n_outliers / len(df) * 100
    })

outliers = pd.DataFrame(outlier_rows)

desc.to_csv(TABLE_DIR / "eda_descritiva.csv")
outliers.to_csv(TABLE_DIR / "eda_outliers_iqr.csv", index=False)

with open(TABLE_DIR / "eda_descritiva.md", "w", encoding="utf-8") as f:
    f.write(desc.to_markdown())

with open(TABLE_DIR / "eda_outliers_iqr.md", "w", encoding="utf-8") as f:
    f.write(outliers.to_markdown(index=False))

print("\n[INFO] Estatística descritiva com skewness:")
print(desc)

print("\n[INFO] Outliers pelo método IQR:")
print(outliers)

# Histogramas individuais.
for col in numeric_cols:
    plt.figure(figsize=(9, 5))
    plt.hist(df[col].dropna(), bins=25)
    plt.title(f"Distribuição de {col}")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"histograma_{col}.png", dpi=160)
    plt.close()

# Boxplots individuais.
for col in numeric_cols:
    plt.figure(figsize=(9, 4))
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f"Boxplot de {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"boxplot_{col}.png", dpi=160)
    plt.close()

# Histograma combinado, útil para comparar escalas visualmente.
df[numeric_cols].hist(figsize=(14, 10), bins=25)
plt.suptitle("Distribuição das Variáveis Numéricas", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "histogramas_variaveis_numericas.png", dpi=160, bbox_inches="tight")
plt.close()

# Boxplot combinado. Como as escalas são muito diferentes, este gráfico mostra por que o escalonamento será necessário.
plt.figure(figsize=(12, 6))
plt.boxplot([df[col].dropna() for col in numeric_cols], vert=False, tick_labels=numeric_cols)
plt.title("Boxplots das Variáveis Numéricas — Escalas Originais")
plt.xlabel("Valor na escala original")
plt.tight_layout()
plt.savefig(FIG_DIR / "boxplots_escalas_originais.png", dpi=160)
plt.close()

# Heatmap de correlação.
corr = df[numeric_cols].corr()
corr.to_csv(TABLE_DIR / "eda_correlacoes.csv")

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Heatmap de Correlação entre Variáveis Numéricas")
plt.tight_layout()
plt.savefig(FIG_DIR / "heatmap_correlacoes.png", dpi=160)
plt.close()

# Pairplot. Pode ser pesado em datasets grandes; aqui é seguro porque o dataset é pequeno.
pairplot_path = FIG_DIR / "pairplot_variaveis.png"
try:
    from pandas.plotting import scatter_matrix
    axes = scatter_matrix(df[numeric_cols], figsize=(14, 14), diagonal="hist", alpha=0.7)
    plt.suptitle("Pairplot das Variáveis Numéricas", y=0.92)
    plt.savefig(pairplot_path, dpi=140, bbox_inches="tight")
    plt.close("all")
    print(f"[OK] Pairplot guardado em: {pairplot_path}")
except Exception as exc:
    print(f"[AVISO] Não foi possível gerar o pairplot: {exc}")

nota = f"""# Notas da Análise Exploratória — Lab 07.1

## Estrutura do dataset

- Linhas: {df.shape[0]}
- Colunas: {df.shape[1]}
- Variáveis numéricas: {", ".join(numeric_cols)}
- Variáveis categóricas: {", ".join(categorical_cols) if categorical_cols else "nenhuma"}

## Observações importantes

1. Este laboratório não tem variável-alvo. A tarefa é descobrir grupos latentes nos dados.
2. As variáveis estão em escalas muito diferentes. Por exemplo, altitude e variação vertical estão em metros, enquanto duração está em minutos.
3. Algoritmos como K-Means, DBSCAN e clustering hierárquico usam distâncias. Sem escalonamento, variáveis de maior escala dominam o cálculo da distância.
4. A presença de outliers pode deslocar centróides no K-Means e alterar a perceção dos grupos.
5. O pairplot e as correlações ajudam a ter uma intuição inicial, mas a decisão final deve usar métricas e interpretação dos perfis.

## Ficheiros gerados

- `outputs/tables/eda_descritiva.csv`
- `outputs/tables/eda_outliers_iqr.csv`
- `outputs/tables/eda_correlacoes.csv`
- `outputs/figures/eda/`
"""

(REPORT_DIR / "NOTAS_EDA.md").write_text(nota, encoding="utf-8")

print("\n[OK] EDA concluída.")
print(f"[OK] Tabelas guardadas em: {TABLE_DIR}")
print(f"[OK] Figuras guardadas em: {FIG_DIR}")
print(f"[OK] Notas guardadas em: {REPORT_DIR / 'NOTAS_EDA.md'}")
