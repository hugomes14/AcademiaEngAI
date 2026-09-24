# 02_preprocessamento.py
# Lab 07.1 — Clustering: Descoberta de Perfis Operacionais de Voo
# Etapa 2: Pré-processamento
#
# Objetivo pedagógico:
# - Preparar os dados para clustering.
# - Codificar variáveis categóricas, caso existam.
# - Tratar valores em falta.
# - Aplicar StandardScaler no dataset completo, porque não há variável-alvo nem split treino/teste.
# - Guardar dados processados e objetos reutilizáveis.

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
DATASET_PATH = BASE_DIR / "data" / "voos_telemetria_completa.csv"

TABLE_DIR = BASE_DIR / "outputs" / "tables"
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 07.1 — ETAPA 2: PRÉ-PROCESSAMENTO")
print("=" * 80)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}")

df_original = pd.read_csv(DATASET_PATH)
df = df_original.copy()

print("\n[INFO] Dataset carregado.")
print(f"[INFO] Dimensão inicial: {df.shape}")

duplicados = int(df.duplicated().sum())
if duplicados > 0:
    print(f"[INFO] Remoção de {duplicados} linhas duplicadas.")
    df = df.drop_duplicates().reset_index(drop=True)
else:
    print("[INFO] Sem duplicados completos.")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\n[INFO] Variáveis numéricas:")
print(numeric_cols)

print("\n[INFO] Variáveis categóricas:")
print(categorical_cols if categorical_cols else "Nenhuma.")

# Tratamento simples e transparente de valores em falta.
for col in numeric_cols:
    n_missing = int(df[col].isna().sum())
    if n_missing > 0:
        mediana = df[col].median()
        print(f"[INFO] {col}: {n_missing} valores em falta preenchidos com mediana {mediana:.4f}.")
        df[col] = df[col].fillna(mediana)

for col in categorical_cols:
    n_missing = int(df[col].isna().sum())
    if n_missing > 0:
        moda = df[col].mode(dropna=True)
        valor = moda.iloc[0] if len(moda) else "Desconhecido"
        print(f"[INFO] {col}: {n_missing} valores em falta preenchidos com moda '{valor}'.")
        df[col] = df[col].fillna(valor)

# Para clustering, não existe alvo.
# Se existirem categóricas, aplica-se one-hot encoding.
# Neste dataset concreto, todas as colunas são numéricas.
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=float)

feature_names = df_encoded.columns.tolist()

print("\n[INFO] Features após encoding:")
print(feature_names)

# Escalonamento obrigatório para modelos baseados em distância.
# Em aprendizagem não supervisionada não existe split treino/teste. O scaler ajusta-se ao dataset completo.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

df_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=df_encoded.index)

df.to_csv(TABLE_DIR / "dados_limpos.csv", index=False)
df_encoded.to_csv(TABLE_DIR / "dados_processados_sem_escala.csv", index=False)
df_scaled.to_csv(TABLE_DIR / "dados_escalonados.csv", index=False)

joblib.dump(scaler, MODEL_DIR / "standard_scaler.pkl")

metadata = {
    "dataset_path": str(DATASET_PATH),
    "n_linhas_original": int(df_original.shape[0]),
    "n_linhas_final": int(df.shape[0]),
    "n_colunas_original": int(df_original.shape[1]),
    "n_features_final": int(len(feature_names)),
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "feature_names": feature_names,
    "scaler": "StandardScaler",
    "nota": "Não existe variável-alvo. O escalonamento foi ajustado ao dataset completo."
}

with open(MODEL_DIR / "metadata_preprocessamento.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

with open(REPORT_DIR / "NOTAS_PREPROCESSAMENTO.md", "w", encoding="utf-8") as f:
    f.write("""# Notas de Pré-processamento — Lab 07.1

## Decisões principais

- Não foi separada nenhuma variável-alvo, porque clustering é aprendizagem não supervisionada.
- Foram detetadas as variáveis numéricas e categóricas de forma automática.
- As variáveis categóricas, se existissem, seriam convertidas por one-hot encoding.
- Foi aplicado `StandardScaler` a todas as features.

## Por que o escalonamento é obrigatório?

K-Means, DBSCAN e clustering hierárquico usam distâncias entre observações. Se uma variável está numa escala muito maior, domina a distância e reduz a influência das restantes variáveis. O escalonamento coloca as features numa escala comparável.

## Artefactos gerados

- `outputs/tables/dados_limpos.csv`
- `outputs/tables/dados_processados_sem_escala.csv`
- `outputs/tables/dados_escalonados.csv`
- `outputs/models/standard_scaler.pkl`
- `outputs/models/metadata_preprocessamento.json`
""")

print("\n[OK] Pré-processamento concluído.")
print(f"[OK] Dados limpos: {TABLE_DIR / 'dados_limpos.csv'}")
print(f"[OK] Dados escalonados: {TABLE_DIR / 'dados_escalonados.csv'}")
print(f"[OK] Scaler: {MODEL_DIR / 'standard_scaler.pkl'}")
