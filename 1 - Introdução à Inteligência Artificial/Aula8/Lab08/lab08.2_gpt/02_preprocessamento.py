# 02_preprocessamento.py
# Lab 08.2 — Algoritmos Genéticos
# Etapa 2: separar treino/teste, escalar features e guardar artefactos.

from pathlib import Path
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
PREP_DIR.mkdir(parents=True, exist_ok=True)

dataset_path = DADOS_DIR / "dados_voo_ga.csv"
target_col = "incidente_reportado"

print("=" * 80)
print("ETAPA 2 — Pré-processamento")
print("=" * 80)

if not dataset_path.exists():
    raise FileNotFoundError("Dataset não encontrado. Corre primeiro o script 01_gerar_dados_e_eda.py.")

df = pd.read_csv(dataset_path)
print(f"Dataset carregado: {dataset_path}")
print(f"Dimensões: {df.shape}")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

feature_names = list(X.columns)

print("\nFeatures usadas:")
for i, col in enumerate(feature_names, start=1):
    print(f"{i:02d}. {col}")

print("\nA separar treino e teste com estratificação...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("Distribuição no treino:")
print(y_train.value_counts(normalize=True).sort_index().round(3))

print("Distribuição no teste:")
print(y_test.value_counts(normalize=True).sort_index().round(3))

print("\nA aplicar StandardScaler.")
print("O scaler aprende média/desvio padrão APENAS no treino para evitar data leakage.")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names,
    index=X_train.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names,
    index=X_test.index,
)

X_train_scaled.to_csv(PREP_DIR / "X_train_scaled.csv", index=False)
X_test_scaled.to_csv(PREP_DIR / "X_test_scaled.csv", index=False)
y_train.to_frame(name=target_col).to_csv(PREP_DIR / "y_train.csv", index=False)
y_test.to_frame(name=target_col).to_csv(PREP_DIR / "y_test.csv", index=False)

with open(PREP_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(PREP_DIR / "feature_names.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=2)

summary = {
    "dataset": str(dataset_path),
    "target": target_col,
    "n_features": len(feature_names),
    "n_train": len(X_train_scaled),
    "n_test": len(X_test_scaled),
    "test_size": 0.25,
    "split": "estratificado",
    "scaler": "StandardScaler",
}
with open(PREP_DIR / "resumo_preprocessamento.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

md = f"""# Resumo do Pré-processamento

- Dataset: `{dataset_path.name}`
- Target: `{target_col}`
- Número de features: {len(feature_names)}
- Linhas treino: {len(X_train_scaled)}
- Linhas teste: {len(X_test_scaled)}
- Split: estratificado
- Escalonamento: `StandardScaler`

## Nota sobre data leakage

O `StandardScaler` foi ajustado apenas no conjunto de treino. Depois, o mesmo transformador foi aplicado ao teste. Isto evita que estatísticas do teste influenciem a preparação dos dados.
"""
(PREP_DIR / "resumo_preprocessamento.md").write_text(md, encoding="utf-8")

print("\nArtefactos guardados em:", PREP_DIR)
print("Etapa 2 concluída com sucesso.")
