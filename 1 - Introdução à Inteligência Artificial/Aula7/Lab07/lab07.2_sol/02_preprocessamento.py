# 02_preprocessamento.py
# Lab 07.2 — Redução de Dimensionalidade
# Etapa 2: separação de X/y, escalonamento e guarda dos objetos.
#
# Nota pedagógica:
# - Este ficheiro corre de forma direta, sem funções e sem main.
# - A coluna tipo_manobra, quando existe, serve para colorir gráficos e para LDA.
# - PCA, Kernel PCA, t-SNE e UMAP devem receber apenas features, não o rótulo.

from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dados" / "sensores_voo_alta_dim.csv"
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
MODELOS_DIR = BASE_DIR / "modelos"
TABELAS_DIR = BASE_DIR / "tabelas"

ARTEFACTOS_DIR.mkdir(exist_ok=True)
MODELOS_DIR.mkdir(exist_ok=True)
TABELAS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("LAB 07.2 — ETAPA 2: PRÉ-PROCESSAMENTO")
print("=" * 80)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
target_col = "tipo_manobra" if "tipo_manobra" in df.columns else None

print(f"Dataset carregado: {DATASET_PATH}")
print(f"Dimensão original: {df.shape}")

if target_col:
    y = df[target_col].copy()
    X_raw = df.drop(columns=[target_col]).copy()
    print(f"Rótulo opcional separado: {target_col}")
else:
    y = pd.Series(["sem_rotulo"] * len(df), name="target_visualizacao")
    X_raw = df.copy()
    print("Sem coluna tipo_manobra. Foi criada uma coluna auxiliar para visualização.")

# Mantém apenas features numéricas no X, pois o dataset do laboratório tem sensores numéricos.
# Se surgirem categóricas, aplica one-hot encoding para manter o pipeline robusto.
numeric_cols = X_raw.select_dtypes(include="number").columns.tolist()
categorical_cols = [col for col in X_raw.columns if col not in numeric_cols]

print(f"Features numéricas: {len(numeric_cols)}")
print(f"Features categóricas detetadas: {categorical_cols}")

if categorical_cols:
    print("Aplicar one-hot encoding às features categóricas não alvo.")
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
else:
    X_encoded = X_raw.copy()

# Substitui valores em falta por mediana nas numéricas e moda nas não numéricas após encoding.
for col in X_encoded.columns:
    if X_encoded[col].isna().any():
        if pd.api.types.is_numeric_dtype(X_encoded[col]):
            X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())
        else:
            X_encoded[col] = X_encoded[col].fillna(X_encoded[col].mode().iloc[0])

feature_names = X_encoded.columns.tolist()

print("\nExplicação crítica:")
print("- PCA procura eixos de maior variância.")
print("- Se um sensor tem escala maior, ele pode dominar artificialmente a análise.")
print("- StandardScaler coloca cada feature com média 0 e desvio padrão 1.")
print("- Como não existe treino/teste neste fluxo não supervisionado, o scaler é ajustado no X completo.")
print("- A coluna tipo_manobra não entra no scaler, para evitar usar o rótulo na redução não supervisionada.")

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_encoded)
X_scaled = pd.DataFrame(X_scaled_array, columns=feature_names, index=df.index)

X_scaled.to_csv(ARTEFACTOS_DIR / "X_scaled.csv", index=False, encoding="utf-8")
pd.DataFrame({"tipo_manobra": y}).to_csv(ARTEFACTOS_DIR / "y.csv", index=False, encoding="utf-8")
X_encoded.to_csv(ARTEFACTOS_DIR / "X_original_features.csv", index=False, encoding="utf-8")
pd.DataFrame({"feature": feature_names}).to_csv(ARTEFACTOS_DIR / "feature_names.csv", index=False, encoding="utf-8")
joblib.dump(scaler, MODELOS_DIR / "scaler_standard.pkl")

metadata = {
    "dataset": str(DATASET_PATH.name),
    "n_linhas": int(df.shape[0]),
    "n_colunas_originais": int(df.shape[1]),
    "target_visualizacao": target_col if target_col else "target_visualizacao",
    "n_features_processadas": len(feature_names),
    "features": feature_names,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "scaler": "StandardScaler",
    "nota": "O rótulo opcional não foi usado para PCA/Kernel PCA/t-SNE/UMAP."
}
(ARTEFACTOS_DIR / "preprocessamento_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

resumo_scaled = X_scaled.describe().T
resumo_scaled.to_csv(TABELAS_DIR / "02_resumo_X_scaled.csv", encoding="utf-8")
resumo_scaled.to_markdown(TABELAS_DIR / "02_resumo_X_scaled.md")

print("\nArtefactos guardados:")
print("- artefactos/X_scaled.csv")
print("- artefactos/y.csv")
print("- artefactos/X_original_features.csv")
print("- modelos/scaler_standard.pkl")
print("- artefactos/preprocessamento_metadata.json")
print("\nResumo após escalonamento:")
print(resumo_scaled[["mean", "std", "min", "max"]].head(10))

print("\nEtapa 2 concluída com sucesso.")
