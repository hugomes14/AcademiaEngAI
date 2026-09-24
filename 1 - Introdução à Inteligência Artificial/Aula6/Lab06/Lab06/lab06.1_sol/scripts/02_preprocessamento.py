# 02_preprocessamento.py
# Lab 06.1 - Regressão — Estimar a Duração de Voo
# Objetivo: separar X/y, criar treino/teste, codificar variáveis e escalar features sem data leakage.

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# -----------------------------
# Configuração de caminhos
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "voos_telemetria.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PROC_DIR = OUTPUT_DIR / "processado"
OBJ_DIR = OUTPUT_DIR / "objetos"
NOTE_DIR = OUTPUT_DIR / "notas"

PROC_DIR.mkdir(parents=True, exist_ok=True)
OBJ_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAME = "duracao_voo_min"
NUMERIC_COLUMNS = ["distancia_planeada", "carga_util_kg", "altitude_media_m"]
CATEGORICAL_COLUMNS = ["condicao_meteo"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
ENCODED_CATEGORICAL_NAME = "condicao_meteo_codigo"
FINAL_FEATURE_NAMES = NUMERIC_COLUMNS + [ENCODED_CATEGORICAL_NAME]

print("=" * 80)
print("02 - PRÉ-PROCESSAMENTO")
print("=" * 80)
print(f"Dataset esperado: {DATASET_PATH}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset não encontrado em {DATASET_PATH}. Corre o orquestrador ou coloca o CSV na pasta data/."
    )

# -----------------------------
# Carregamento e validação
# -----------------------------
df = pd.read_csv(DATASET_PATH)
print(f"\nDataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")

required_columns = FEATURE_COLUMNS + [TARGET_NAME]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Colunas em falta no dataset: {missing_columns}")

if df[required_columns].isna().sum().sum() > 0:
    print("\nAviso: existem valores em falta. Será aplicada uma estratégia simples de imputação.")
    for column in NUMERIC_COLUMNS:
        df[column] = df[column].fillna(df[column].median())
    df["condicao_meteo"] = df["condicao_meteo"].fillna(df["condicao_meteo"].mode(dropna=True)[0])
    df[TARGET_NAME] = df[TARGET_NAME].fillna(df[TARGET_NAME].median())
else:
    print("\nNão foram encontrados valores em falta nas colunas usadas pelo modelo.")

# -----------------------------
# Separação X/y
# -----------------------------
X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_NAME].copy()

print("\nFeatures usadas:")
print(FEATURE_COLUMNS)
print(f"Variável-alvo: {TARGET_NAME}")

# -----------------------------
# Train/test split simples
# -----------------------------
# Não usamos estratificação por defeito porque a variável-alvo é contínua.
# Em regressão, estratificar exige discretizar o alvo, o que pode introduzir complexidade desnecessária neste laboratório.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(f"\nTreino: {X_train_raw.shape[0]} linhas")
print(f"Teste: {X_test_raw.shape[0]} linhas")

# -----------------------------
# Encoding ordinal da meteorologia
# -----------------------------
# A variável condicao_meteo tem uma ordem natural:
# Bom < Moderado < Adverso.
# Por isso usamos OrdinalEncoder, em vez de one-hot encoding.
# Se surgisse uma categórica sem ordem natural, a opção recomendada seria OneHotEncoder.
weather_categories = [["Bom", "Moderado", "Adverso"]]
ordinal_encoder = OrdinalEncoder(categories=weather_categories, handle_unknown="use_encoded_value", unknown_value=-1)

X_train_encoded = X_train_raw.copy()
X_test_encoded = X_test_raw.copy()

X_train_encoded[[ENCODED_CATEGORICAL_NAME]] = ordinal_encoder.fit_transform(X_train_raw[["condicao_meteo"]])
X_test_encoded[[ENCODED_CATEGORICAL_NAME]] = ordinal_encoder.transform(X_test_raw[["condicao_meteo"]])

X_train_encoded = X_train_encoded.drop(columns=CATEGORICAL_COLUMNS)
X_test_encoded = X_test_encoded.drop(columns=CATEGORICAL_COLUMNS)

# Garantimos a mesma ordem de colunas no treino e no teste.
X_train_encoded = X_train_encoded[FINAL_FEATURE_NAMES]
X_test_encoded = X_test_encoded[FINAL_FEATURE_NAMES]

print("\nExemplo de features após encoding ordinal:")
print(X_train_encoded.head())

# -----------------------------
# Escalonamento sem data leakage
# -----------------------------
# O scaler é ajustado apenas no conjunto de treino.
# Depois é aplicado ao treino e ao teste.
# Isto evita que estatísticas do teste entrem no treino, o que seria data leakage.
scaler = StandardScaler()
X_train_scaled_array = scaler.fit_transform(X_train_encoded)
X_test_scaled_array = scaler.transform(X_test_encoded)

X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=FINAL_FEATURE_NAMES, index=X_train_encoded.index)
X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=FINAL_FEATURE_NAMES, index=X_test_encoded.index)

print("\nMédias das features escaladas no treino, aproximadamente 0:")
print(X_train_scaled.mean().round(6))
print("\nDesvios-padrão das features escaladas no treino, aproximadamente 1:")
print(X_train_scaled.std(ddof=0).round(6))

# -----------------------------
# Guarda dos conjuntos processados
# -----------------------------
X_train_raw.to_csv(PROC_DIR / "X_train_raw.csv", index=True)
X_test_raw.to_csv(PROC_DIR / "X_test_raw.csv", index=True)
X_train_encoded.to_csv(PROC_DIR / "X_train_sem_escala.csv", index=True)
X_test_encoded.to_csv(PROC_DIR / "X_test_sem_escala.csv", index=True)
X_train_scaled.to_csv(PROC_DIR / "X_train_escalado.csv", index=True)
X_test_scaled.to_csv(PROC_DIR / "X_test_escalado.csv", index=True)
y_train.to_frame(TARGET_NAME).to_csv(PROC_DIR / "y_train.csv", index=True)
y_test.to_frame(TARGET_NAME).to_csv(PROC_DIR / "y_test.csv", index=True)

with open(OBJ_DIR / "preprocessamento.pkl", "wb") as file:
    pickle.dump({
        "ordinal_encoder": ordinal_encoder,
        "scaler": scaler,
        "target_name": TARGET_NAME,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "encoded_categorical_name": ENCODED_CATEGORICAL_NAME,
        "final_feature_names": FINAL_FEATURE_NAMES,
        "weather_categories": weather_categories,
        "test_size": 0.20,
        "random_state": 42
    }, file)

notes = []
notes.append("# Notas do Pré-processamento\n")
notes.append("- O dataset foi separado em treino e teste com `test_size=0.20` e `random_state=42`.\n")
notes.append("- Não foi usada estratificação porque a variável-alvo é contínua.\n")
notes.append("- A variável `condicao_meteo` foi codificada de forma ordinal: Bom=0, Moderado=1, Adverso=2.\n")
notes.append("- O `StandardScaler` foi ajustado apenas no treino e aplicado depois ao treino e ao teste.\n")
notes.append("- Esta regra evita data leakage, porque o teste simula dados futuros desconhecidos durante o treino.\n")
notes.append("- Os modelos lineares sem regularização podem ser lidos com features sem escala; Ridge, Lasso e Polinomial usam features escaladas.\n")

(NOTE_DIR / "02_notas_preprocessamento.md").write_text("".join(notes), encoding="utf-8")

print("\nFicheiros gerados:")
print(f"- Dados processados: {PROC_DIR}")
print(f"- Objetos: {OBJ_DIR / 'preprocessamento.pkl'}")
print(f"- Notas: {NOTE_DIR / '02_notas_preprocessamento.md'}")
print("\n02 - Pré-processamento concluído com sucesso.")
