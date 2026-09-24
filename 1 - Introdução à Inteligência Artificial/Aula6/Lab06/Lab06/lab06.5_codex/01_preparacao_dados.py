# -*- coding: utf-8 -*-
"""
Lab 06.5 - Preparação dos dados para Regressão com Rede Neural de 1 neurónio.

Este script:
- carrega o dataset de telemetria de voos;
- faz validações simples de qualidade;
- aplica one-hot encoding à variável categórica;
- padroniza as features numéricas/codificadas;
- divide os dados em treino, validação e teste: 70% / 15% / 15%;
- guarda objetos e conjuntos processados para os scripts seguintes.

Nota pedagógica:
O escalonamento é crítico em redes neurais porque features com escalas muito diferentes
podem tornar o processo de otimização lento ou instável.
"""

from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dados" / "voos_telemetria.csv"
DIR_PREP = RAIZ / "artefactos" / "preprocessamento"
DIR_IMG = RAIZ / "imagens"

DIR_PREP.mkdir(parents=True, exist_ok=True)
DIR_IMG.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.5 - PREPARAÇÃO DOS DADOS")
print("=" * 80)

if not DADOS.exists():
    raise FileNotFoundError(
        f"Dataset não encontrado: {DADOS}\n"
        "Coloca o ficheiro voos_telemetria.csv dentro da pasta dados/."
    )

# -----------------------------------------------------------------------------
# 1) Carregamento e inspeção inicial
# -----------------------------------------------------------------------------
df = pd.read_csv(DADOS)
print(f"\nDataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
print("\nPrimeiras linhas:")
print(df.head())
print("\nTipos de dados:")
print(df.dtypes)

colunas_esperadas = [
    "distancia_planeada",
    "carga_util_kg",
    "altitude_media_m",
    "condicao_meteo",
    "duracao_voo_min",
]
colunas_em_falta = [col for col in colunas_esperadas if col not in df.columns]
if colunas_em_falta:
    raise ValueError(f"Colunas em falta no dataset: {colunas_em_falta}")

print("\nValores em falta por coluna:")
print(df.isna().sum())

linhas_antes = len(df)
df = df.drop_duplicates().dropna().copy()
linhas_depois = len(df)
print(f"\nLinhas removidas por duplicados/valores em falta: {linhas_antes - linhas_depois}")

# -----------------------------------------------------------------------------
# 2) Análise rápida da variável alvo
# -----------------------------------------------------------------------------
target = "duracao_voo_min"
features_numericas = ["distancia_planeada", "carga_util_kg", "altitude_media_m"]
features_categoricas = ["condicao_meteo"]

print("\nResumo estatístico da variável alvo:")
print(df[target].describe())
print(f"Skewness da duração do voo: {df[target].skew():.4f}")

plt.figure(figsize=(9, 5))
plt.hist(df[target], bins=30, edgecolor="black", alpha=0.85)
plt.title("Distribuição da duração do voo")
plt.xlabel("Duração do voo (min)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(DIR_IMG / "01_distribuicao_duracao_voo.png", dpi=160)
plt.savefig(DIR_IMG / "01_distribuicao_duracao_voo.pdf")
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot(df[target], vert=False)
plt.title("Boxplot da duração do voo")
plt.xlabel("Duração do voo (min)")
plt.tight_layout()
plt.savefig(DIR_IMG / "01_boxplot_duracao_voo.png", dpi=160)
plt.savefig(DIR_IMG / "01_boxplot_duracao_voo.pdf")
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[features_numericas + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre variáveis numéricas e duração do voo")
plt.tight_layout()
plt.savefig(DIR_IMG / "01_correlacoes_numericas.png", dpi=160)
plt.savefig(DIR_IMG / "01_correlacoes_numericas.pdf")
plt.close()

# -----------------------------------------------------------------------------
# 3) Separação X/y e split 70% / 15% / 15%
# -----------------------------------------------------------------------------
X = df[features_numericas + features_categoricas].copy()
y = df[target].copy()

# Primeiro split: 70% treino e 30% temporário.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Segundo split: os 30% temporários passam a 15% validação e 15% teste.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("\nDimensões dos conjuntos:")
print(f"Treino:    X={X_train.shape}, y={y_train.shape}")
print(f"Validação: X={X_val.shape}, y={y_val.shape}")
print(f"Teste:     X={X_test.shape}, y={y_test.shape}")

# -----------------------------------------------------------------------------
# 4) Pré-processamento: one-hot + scaling
# -----------------------------------------------------------------------------
# O OneHotEncoder usa drop='first' para evitar colinearidade perfeita entre dummies.
# Isto torna a comparação de coeficientes entre PyTorch e sklearn mais estável.
onehot = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
preprocessador = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), features_numericas),
        ("cat", onehot, features_categoricas),
    ],
    remainder="drop",
)

# Muito importante: o fit é feito APENAS no treino.
# Se o scaler/encoder fossem ajustados com validação ou teste, existiria data leakage.
X_train_proc = preprocessador.fit_transform(X_train)
X_val_proc = preprocessador.transform(X_val)
X_test_proc = preprocessador.transform(X_test)

try:
    nomes_features = preprocessador.get_feature_names_out()
    nomes_features = [nome.replace("num__", "").replace("cat__", "") for nome in nomes_features]
except Exception:
    nomes_features = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

print("\nFeatures após pré-processamento:")
for i, nome in enumerate(nomes_features, start=1):
    print(f"  {i:02d}. {nome}")

# -----------------------------------------------------------------------------
# 5) Guardar artefactos
# -----------------------------------------------------------------------------
pacote = {
    "X_train": X_train,
    "X_val": X_val,
    "X_test": X_test,
    "y_train": y_train,
    "y_val": y_val,
    "y_test": y_test,
    "X_train_proc": X_train_proc,
    "X_val_proc": X_val_proc,
    "X_test_proc": X_test_proc,
    "y_train_array": y_train.to_numpy(dtype=np.float32),
    "y_val_array": y_val.to_numpy(dtype=np.float32),
    "y_test_array": y_test.to_numpy(dtype=np.float32),
    "feature_names": nomes_features,
    "target_name": target,
    "numeric_features": features_numericas,
    "categorical_features": features_categoricas,
}

with open(DIR_PREP / "dados_processados.pkl", "wb") as f:
    pickle.dump(pacote, f)

with open(DIR_PREP / "preprocessador.pkl", "wb") as f:
    pickle.dump(preprocessador, f)

pd.DataFrame(X_train_proc, columns=nomes_features).assign(duracao_voo_min=y_train.to_numpy()).to_csv(
    DIR_PREP / "treino_processado.csv", index=False
)
pd.DataFrame(X_val_proc, columns=nomes_features).assign(duracao_voo_min=y_val.to_numpy()).to_csv(
    DIR_PREP / "validacao_processado.csv", index=False
)
pd.DataFrame(X_test_proc, columns=nomes_features).assign(duracao_voo_min=y_test.to_numpy()).to_csv(
    DIR_PREP / "teste_processado.csv", index=False
)

print("\nArtefactos guardados em:")
print(f"- {DIR_PREP / 'dados_processados.pkl'}")
print(f"- {DIR_PREP / 'preprocessador.pkl'}")
print("\nPreparação concluída com sucesso.")
