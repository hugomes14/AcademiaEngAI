# 02_preprocessamento.py
# Lab 06.4 — Modelos Ensemble
# Etapa: Pré-processamento
#
# Este script prepara os dados para treino:
# - Separa X e y.
# - Faz split estratificado para respeitar o desbalanceamento.
# - Aplica encoding ordinal à turbulência.
# - Aplica one-hot encoding ao tipo de missão.
# - Escalona variáveis numéricas.
#
# Nota sobre data leakage:
# O pré-processador faz fit apenas no treino. O teste usa apenas transform.
# Isto impede que informação do conjunto de teste contamine o treino.

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dados" / "voos_pre_voo.csv"
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"

PREP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.4 — ETAPA 2: PRÉ-PROCESSAMENTO")
print("=" * 80)

df = pd.read_csv(DATASET_PATH)
print(f"\nDataset lido com dimensão: {df.shape}")

target = "incidente_reportado"
colunas_numericas = [
    "idade_aeronave_anos",
    "horas_voo_desde_ultima_manutencao",
    "experiencia_piloto_anos",
]
coluna_ordinal = ["previsao_turbulencia"]
coluna_nominal = ["tipo_missao"]

X = df.drop(columns=[target])
y = df[target].astype(int)

print("\nDistribuição do alvo antes do split:")
print(y.value_counts(normalize=True).sort_index().mul(100).round(2))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nDistribuição do alvo no treino:")
print(y_train.value_counts(normalize=True).sort_index().mul(100).round(2))

print("\nDistribuição do alvo no teste:")
print(y_test.value_counts(normalize=True).sort_index().mul(100).round(2))

preprocessador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            Pipeline(steps=[
                ("scaler", StandardScaler()),
            ]),
            colunas_numericas,
        ),
        (
            "turbulencia_ordinal",
            OrdinalEncoder(categories=[["Baixa", "Média", "Alta"]]),
            coluna_ordinal,
        ),
        (
            "missao_onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            coluna_nominal,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

print("\nA ajustar o pré-processador apenas no conjunto de treino...")
X_train_array = preprocessador.fit_transform(X_train)
X_test_array = preprocessador.transform(X_test)

feature_names = list(preprocessador.get_feature_names_out())

X_train_proc = pd.DataFrame(X_train_array, columns=feature_names, index=X_train.index)
X_test_proc = pd.DataFrame(X_test_array, columns=feature_names, index=X_test.index)

print("\nFeatures após pré-processamento:")
for nome in feature_names:
    print(f"- {nome}")

print(f"\nDimensão X_train processado: {X_train_proc.shape}")
print(f"Dimensão X_test processado: {X_test_proc.shape}")

joblib.dump(preprocessador, PREP_DIR / "preprocessador.pkl")
joblib.dump(
    {
        "X_train": X_train_proc,
        "X_test": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_original": X_train,
        "X_test_original": X_test,
    },
    PREP_DIR / "dados_processados.pkl",
)

X_train_proc.to_csv(PREP_DIR / "X_train_processado.csv", index=True)
X_test_proc.to_csv(PREP_DIR / "X_test_processado.csv", index=True)
y_train.to_csv(PREP_DIR / "y_train.csv", index=True)
y_test.to_csv(PREP_DIR / "y_test.csv", index=True)

metadata = {
    "target": target,
    "colunas_numericas": colunas_numericas,
    "coluna_ordinal": coluna_ordinal,
    "coluna_nominal": coluna_nominal,
    "ordem_turbulencia": ["Baixa", "Média", "Alta"],
    "feature_names": feature_names,
    "test_size": 0.20,
    "random_state": 42,
    "split": "estratificado",
}
(PREP_DIR / "metadata_preprocessamento.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

notas = """# Notas de Pré-processamento

## Encoding

- `previsao_turbulencia` foi codificada de forma ordinal: Baixa < Média < Alta.
- `tipo_missao` foi codificada com one-hot encoding porque não existe ordem natural.

## Escalonamento

As variáveis numéricas foram escalonadas com `StandardScaler`.

Embora árvores e ensembles baseados em árvores não exijam escalonamento, esta etapa
mantém o pipeline consistente e evita problemas se forem adicionados modelos sensíveis
à escala.

## Data Leakage

O `fit` do pré-processador foi aplicado apenas ao conjunto de treino. O conjunto de teste
só recebeu `transform`. Isto evita que informação estatística do teste entre no treino.

## Desbalanceamento

O split foi estratificado para manter proporções semelhantes de classes no treino e no teste.
"""
(PREP_DIR / "notas_preprocessamento.md").write_text(notas, encoding="utf-8")

print("\nArtefactos de pré-processamento guardados em:")
print(PREP_DIR)
print("\nEtapa 2 concluída com sucesso.")
