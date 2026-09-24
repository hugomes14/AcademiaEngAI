# -*- coding: utf-8 -*-
"""
Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos
Etapa 02: Pré-processamento

Objetivo:
- Separar X e y.
- Aplicar split treino/teste estratificado para preservar o desbalanceamento.
- Codificar variáveis categóricas.
- Escalonar variáveis sensíveis à escala sem provocar data leakage.
- Guardar dados processados e objetos de transformação.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

print("=" * 80)
print("ETAPA 02 — PRÉ-PROCESSAMENTO")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "voos_pre_voo.csv"
DATA_DIR = BASE_DIR / "artefactos" / "dados"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "incidente_reportado"
NUMERIC_COLS = [
    "idade_aeronave_anos",
    "horas_voo_desde_ultima_manutencao",
    "experiencia_piloto_anos",
]
ORDINAL_COLS = ["previsao_turbulencia"]
NOMINAL_COLS = ["tipo_missao"]

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

expected_cols = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS + [TARGET]
missing_cols = [col for col in expected_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Faltam colunas obrigatórias no dataset: {missing_cols}")

print("\nResumo antes do pré-processamento:")
print(df.info())

# Remoção defensiva de linhas com valores em falta.
# Num projeto real, poderíamos aplicar imputação. Aqui, a estratégia é explícita e simples.
missing_total = int(df[expected_cols].isna().sum().sum())
if missing_total > 0:
    print(f"\nForam encontrados {missing_total} valores em falta. As linhas afetadas serão removidas.")
    df = df.dropna(subset=expected_cols).copy()
else:
    print("\nNão foram encontrados valores em falta nas colunas usadas pelo modelo.")

# Separação de preditores e alvo.
X = df[NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS].copy()
y = df[TARGET].astype(int).copy()

print("\nDistribuição global do alvo:")
print(y.value_counts(normalize=True).sort_index().mul(100).round(2).astype(str) + "%")

# Split estratificado:
# Como a classe 1 é minoritária, o stratify=y preserva proporções parecidas em treino e teste.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("\nDistribuição no treino:")
print(y_train.value_counts(normalize=True).sort_index().mul(100).round(2).astype(str) + "%")
print("\nDistribuição no teste:")
print(y_test.value_counts(normalize=True).sort_index().mul(100).round(2).astype(str) + "%")

# Pré-processador principal, com escalonamento.
# Atenção a data leakage: o fit ocorre apenas sobre X_train.
# O X_test recebe apenas transform, sem influenciar médias, desvios, categorias ou codificações.
preprocessor_scaled = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        (
            "turbulencia_ord",
            Pipeline(
                steps=[
                    ("ordinal", OrdinalEncoder(categories=[["Baixa", "Média", "Alta"]], handle_unknown="use_encoded_value", unknown_value=-1)),
                    ("scaler", StandardScaler()),
                ]
            ),
            ORDINAL_COLS,
        ),
        ("missao_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# Pré-processador sem escalonamento, usado apenas para demonstrar o impacto da escala em KNN/SVM.
preprocessor_unscaled = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_COLS),
        ("turbulencia_ord", OrdinalEncoder(categories=[["Baixa", "Média", "Alta"]], handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_COLS),
        ("missao_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

print("\nAjuste do pré-processador com escalonamento apenas no treino...")
X_train_scaled_array = preprocessor_scaled.fit_transform(X_train)
X_test_scaled_array = preprocessor_scaled.transform(X_test)

print("Ajuste do pré-processador sem escalonamento apenas no treino...")
X_train_unscaled_array = preprocessor_unscaled.fit_transform(X_train)
X_test_unscaled_array = preprocessor_unscaled.transform(X_test)

feature_names_scaled = list(preprocessor_scaled.get_feature_names_out())
feature_names_unscaled = list(preprocessor_unscaled.get_feature_names_out())

X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=feature_names_scaled, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=feature_names_scaled, index=X_test.index)
X_train_unscaled = pd.DataFrame(X_train_unscaled_array, columns=feature_names_unscaled, index=X_train.index)
X_test_unscaled = pd.DataFrame(X_test_unscaled_array, columns=feature_names_unscaled, index=X_test.index)

# Guardar dados processados.
X_train_scaled.to_csv(DATA_DIR / "X_train_scaled.csv", index=False, encoding="utf-8")
X_test_scaled.to_csv(DATA_DIR / "X_test_scaled.csv", index=False, encoding="utf-8")
X_train_unscaled.to_csv(DATA_DIR / "X_train_unscaled.csv", index=False, encoding="utf-8")
X_test_unscaled.to_csv(DATA_DIR / "X_test_unscaled.csv", index=False, encoding="utf-8")
y_train.to_frame(name=TARGET).to_csv(DATA_DIR / "y_train.csv", index=False, encoding="utf-8")
y_test.to_frame(name=TARGET).to_csv(DATA_DIR / "y_test.csv", index=False, encoding="utf-8")

# Guardar também os dados originais separados, úteis para auditoria e interpretação.
X_train.to_csv(DATA_DIR / "X_train_original.csv", index=False, encoding="utf-8")
X_test.to_csv(DATA_DIR / "X_test_original.csv", index=False, encoding="utf-8")

# Guardar objetos reutilizáveis.
joblib.dump(preprocessor_scaled, MODEL_DIR / "preprocessor_scaled.pkl")
joblib.dump(preprocessor_unscaled, MODEL_DIR / "preprocessor_unscaled.pkl")

metadata = {
    "dataset": str(DATASET_PATH.relative_to(BASE_DIR)),
    "target": TARGET,
    "numeric_cols": NUMERIC_COLS,
    "ordinal_cols": ORDINAL_COLS,
    "nominal_cols": NOMINAL_COLS,
    "feature_names_scaled": feature_names_scaled,
    "feature_names_unscaled": feature_names_unscaled,
    "test_size": 0.25,
    "random_state": 42,
    "stratified_split": True,
    "class_distribution_train": y_train.value_counts().sort_index().to_dict(),
    "class_distribution_test": y_test.value_counts().sort_index().to_dict(),
}
(DATA_DIR / "metadata_pre_processamento.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nFicheiros guardados em:")
print(DATA_DIR)
print("\nObjetos guardados em:")
print(MODEL_DIR)
print("\nEtapa 02 concluída com sucesso.")
