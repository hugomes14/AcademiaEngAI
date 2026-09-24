# 01_gerar_dados_e_eda.py
# Lab 08.2 — Algoritmos Genéticos
# Etapa 1: gerar dataset sintético e fazer análise exploratória inicial.

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
EDA_DIR = BASE_DIR / "artefactos" / "eda"
DADOS_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

dataset_path = DADOS_DIR / "dados_voo_ga.csv"

print("=" * 80)
print("ETAPA 1 — Geração de dados sintéticos e EDA")
print("=" * 80)

if not dataset_path.exists():
    print("Dataset não encontrado. A gerar dados sintéticos de voos...")

    X, y = make_classification(
        n_samples=600,
        n_features=18,
        n_informative=6,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.82, 0.18],
        class_sep=1.15,
        flip_y=0.025,
        random_state=SEED,
    )

    feature_names = [f"sensor_{i:02d}" for i in range(1, 19)]
    df = pd.DataFrame(X, columns=feature_names)

    # Criação de algumas variáveis com nomes mais próximos do domínio do laboratório.
    df["idade_aeronave_anos"] = np.clip(12 + 4 * df["sensor_01"] + np.random.normal(0, 1.2, len(df)), 1, 35)
    df["horas_desde_manutencao"] = np.clip(90 + 25 * df["sensor_02"] + np.random.normal(0, 8, len(df)), 1, 260)
    df["stress_operacional"] = np.clip(50 + 12 * df["sensor_03"] + 8 * df["sensor_04"], 0, 100)

    # Remove três sensores para manter dimensão moderada e acrescenta variáveis de domínio.
    df = df.drop(columns=["sensor_01", "sensor_02", "sensor_03"])
    df["incidente_reportado"] = y.astype(int)

    df.to_csv(dataset_path, index=False)
    print(f"Dataset criado em: {dataset_path}")
else:
    print(f"Dataset já existe: {dataset_path}")

df = pd.read_csv(dataset_path)
print("\nPrimeiras linhas:")
print(df.head())

print("\nDimensões do dataset:")
print(df.shape)

print("\nTipos de dados:")
print(df.dtypes)

target_col = "incidente_reportado"
feature_cols = [c for c in df.columns if c != target_col]

print("\nDistribuição da variável-alvo:")
class_counts = df[target_col].value_counts().sort_index()
class_percent = df[target_col].value_counts(normalize=True).sort_index() * 100
target_summary = pd.DataFrame({"contagem": class_counts, "percentagem": class_percent.round(2)})
print(target_summary)
target_summary.to_csv(EDA_DIR / "distribuicao_target.csv")

summary = df[feature_cols].describe().T
summary["skewness"] = df[feature_cols].skew()
summary.to_csv(EDA_DIR / "estatisticas_features.csv")
summary.to_markdown(EDA_DIR / "estatisticas_features.md")

print("\nResumo estatístico das features:")
print(summary.head(10))

plt.figure(figsize=(7, 4))
plt.bar(target_summary.index.astype(str), target_summary["contagem"])
plt.title("Distribuição da variável-alvo")
plt.xlabel("Incidente reportado")
plt.ylabel("Contagem")
plt.tight_layout()
plt.savefig(EDA_DIR / "distribuicao_target.png", dpi=160)
plt.savefig(EDA_DIR / "distribuicao_target.pdf")
plt.close()

selected_features = feature_cols[:12]
df[selected_features].hist(figsize=(14, 10), bins=25)
plt.suptitle("Distribuição das primeiras features", y=1.02)
plt.tight_layout()
plt.savefig(EDA_DIR / "histogramas_features.png", dpi=160)
plt.savefig(EDA_DIR / "histogramas_features.pdf")
plt.close()

plt.figure(figsize=(14, 8))
plt.boxplot([df[col].dropna().values for col in selected_features], vert=False, labels=selected_features)
plt.title("Boxplots das primeiras features")
plt.tight_layout()
plt.savefig(EDA_DIR / "boxplots_features.png", dpi=160)
plt.savefig(EDA_DIR / "boxplots_features.pdf")
plt.close()

corr_cols = feature_cols[:18] + [target_col]
corr = df[corr_cols].corr(numeric_only=True)
plt.figure(figsize=(12, 10))
img = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(img, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Mapa de correlação")
plt.tight_layout()
plt.savefig(EDA_DIR / "heatmap_correlacoes.png", dpi=160)
plt.savefig(EDA_DIR / "heatmap_correlacoes.pdf")
plt.close()

notes = f"""# Notas da EDA

## Dataset
- Ficheiro: `{dataset_path.name}`
- Linhas: {df.shape[0]}
- Colunas: {df.shape[1]}
- Variável-alvo: `{target_col}`

## Distribuição da variável-alvo

{target_summary.to_markdown()}

## Observações pedagógicas
- O dataset simula uma tarefa de classificação binária.
- A seleção de features por Algoritmo Genético é útil porque existe um conjunto relativamente grande de variáveis.
- Algumas features podem ser redundantes, ruidosas ou pouco relevantes.
- O AG vai tentar encontrar um subconjunto com bom desempenho preditivo sem usar todas as features.
- A avaliação da fitness deve usar validação cruzada para reduzir o risco de sobreajuste ao treino.
"""
(EDA_DIR / "notas_eda.md").write_text(notes, encoding="utf-8")

print("\nArtefactos de EDA guardados em:", EDA_DIR)
print("Etapa 1 concluída com sucesso.")
