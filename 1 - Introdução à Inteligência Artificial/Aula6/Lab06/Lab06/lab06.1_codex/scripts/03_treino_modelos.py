# 03_treino_modelos.py
# Lab 06.1 - Regressão — Estimar a Duração de Voo
# Objetivo: treinar modelos de regressão e guardar modelos, previsões, tempos e coeficientes.
# Nota: este ficheiro não calcula métricas; essa responsabilidade fica no script 04.

from pathlib import Path
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# -----------------------------
# Configuração de caminhos
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
PROC_DIR = OUTPUT_DIR / "processado"
MODEL_DIR = OUTPUT_DIR / "modelos"
PRED_DIR = OUTPUT_DIR / "previsoes"
TAB_DIR = OUTPUT_DIR / "tabelas"
NOTE_DIR = OUTPUT_DIR / "notas"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAME = "duracao_voo_min"
FEATURE_NAMES = ["distancia_planeada", "carga_util_kg", "altitude_media_m", "condicao_meteo_codigo"]
SIMPLE_FEATURE = ["distancia_planeada"]

print("=" * 80)
print("03 - TREINO DE MODELOS")
print("=" * 80)

required_files = [
    PROC_DIR / "X_train_sem_escala.csv",
    PROC_DIR / "X_test_sem_escala.csv",
    PROC_DIR / "X_train_escalado.csv",
    PROC_DIR / "X_test_escalado.csv",
    PROC_DIR / "y_train.csv",
    PROC_DIR / "y_test.csv"
]
for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Ficheiro necessário não encontrado: {file_path}. Corre primeiro o script 02.")

# -----------------------------
# Carregamento dos dados processados
# -----------------------------
X_train_unscaled = pd.read_csv(PROC_DIR / "X_train_sem_escala.csv", index_col=0)
X_test_unscaled = pd.read_csv(PROC_DIR / "X_test_sem_escala.csv", index_col=0)
X_train_scaled = pd.read_csv(PROC_DIR / "X_train_escalado.csv", index_col=0)
X_test_scaled = pd.read_csv(PROC_DIR / "X_test_escalado.csv", index_col=0)
y_train = pd.read_csv(PROC_DIR / "y_train.csv", index_col=0)[TARGET_NAME]
y_test = pd.read_csv(PROC_DIR / "y_test.csv", index_col=0)[TARGET_NAME]

print(f"\nX_train sem escala: {X_train_unscaled.shape}")
print(f"X_train escalado: {X_train_scaled.shape}")
print(f"y_train: {y_train.shape}")

# -----------------------------
# Definição dos modelos
# -----------------------------
# Modelo simples: usa só a distância planeada.
# Modelo múltiplo: usa todas as features codificadas.
# Ridge e Lasso: usam regularização; por isso treinam sobre features escaladas.
# Polinomial grau 2: cria termos quadráticos e interações para capturar não linearidades.
model_specs = [
    {
        "nome": "regressao_linear_simples_distancia",
        "modelo": LinearRegression(),
        "X_train": X_train_unscaled[SIMPLE_FEATURE],
        "X_test": X_test_unscaled[SIMPLE_FEATURE],
        "features": SIMPLE_FEATURE,
        "usa_escala": False,
        "descricao": "Modelo interpretável que usa apenas a distância planeada."
    },
    {
        "nome": "regressao_linear_multipla",
        "modelo": LinearRegression(),
        "X_train": X_train_unscaled[FEATURE_NAMES],
        "X_test": X_test_unscaled[FEATURE_NAMES],
        "features": FEATURE_NAMES,
        "usa_escala": False,
        "descricao": "Modelo linear com todas as variáveis disponíveis antes da descolagem."
    },
    {
        "nome": "ridge",
        "modelo": Ridge(alpha=1.0, random_state=42),
        "X_train": X_train_scaled[FEATURE_NAMES],
        "X_test": X_test_scaled[FEATURE_NAMES],
        "features": FEATURE_NAMES,
        "usa_escala": True,
        "descricao": "Modelo linear regularizado L2; reduz coeficientes extremos e ajuda na estabilidade."
    },
    {
        "nome": "lasso",
        "modelo": Lasso(alpha=0.05, max_iter=10000, random_state=42),
        "X_train": X_train_scaled[FEATURE_NAMES],
        "X_test": X_test_scaled[FEATURE_NAMES],
        "features": FEATURE_NAMES,
        "usa_escala": True,
        "descricao": "Modelo linear regularizado L1; pode colocar coeficientes pouco úteis a zero."
    },
    {
        "nome": "regressao_polinomial_grau_2",
        "modelo": Pipeline([
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear_regression", LinearRegression())
        ]),
        "X_train": X_train_scaled[FEATURE_NAMES],
        "X_test": X_test_scaled[FEATURE_NAMES],
        "features": FEATURE_NAMES,
        "usa_escala": True,
        "descricao": "Modelo com termos de grau 2 para capturar relações não lineares e interações."
    }
]

training_rows = []
coefficient_rows = []
all_predictions = pd.DataFrame({
    "indice_original": y_test.index,
    "y_real": y_test.values
})

# -----------------------------
# Treino e guarda de cada modelo
# -----------------------------
for spec in model_specs:
    model_name = spec["nome"]
    model = spec["modelo"]
    X_train_model = spec["X_train"]
    X_test_model = spec["X_test"]

    print("\n" + "-" * 80)
    print(f"A treinar: {model_name}")
    print(spec["descricao"])
    print(f"Features usadas: {spec['features']}")
    print(f"Usa escala: {spec['usa_escala']}")

    start_time = time.perf_counter()
    model.fit(X_train_model, y_train)
    elapsed_seconds = time.perf_counter() - start_time

    predictions = model.predict(X_test_model)

    with open(MODEL_DIR / f"{model_name}.pkl", "wb") as file:
        pickle.dump({
            "modelo": model,
            "nome": model_name,
            "features": spec["features"],
            "usa_escala": spec["usa_escala"],
            "descricao": spec["descricao"]
        }, file)

    pred_df = pd.DataFrame({
        "indice_original": y_test.index,
        "y_real": y_test.values,
        "y_previsto": predictions
    })
    pred_df.to_csv(PRED_DIR / f"previsoes_{model_name}.csv", index=False)
    all_predictions[model_name] = predictions

    training_rows.append({
        "modelo": model_name,
        "tempo_treino_segundos": elapsed_seconds,
        "n_linhas_treino": X_train_model.shape[0],
        "n_features": X_train_model.shape[1],
        "usa_escala": spec["usa_escala"],
        "descricao": spec["descricao"]
    })

    print(f"Tempo de treino: {elapsed_seconds:.6f} segundos")
    print(f"Modelo guardado em: {MODEL_DIR / (model_name + '.pkl')}")
    print(f"Previsões guardadas em: {PRED_DIR / ('previsoes_' + model_name + '.csv')}")

    # Coeficientes dos modelos lineares diretamente interpretáveis.
    if model_name == "regressao_linear_multipla":
        for feature_name, coefficient in zip(spec["features"], model.coef_):
            coefficient_rows.append({
                "modelo": model_name,
                "feature": feature_name,
                "coeficiente": coefficient,
                "nota": "Coeficiente em unidades originais da feature, porque o modelo foi treinado sem escala."
            })
        coefficient_rows.append({
            "modelo": model_name,
            "feature": "interceto",
            "coeficiente": model.intercept_,
            "nota": "Valor base estimado quando as features são zero; pode não ter interpretação física direta."
        })

    if model_name == "lasso":
        for feature_name, coefficient in zip(spec["features"], model.coef_):
            coefficient_rows.append({
                "modelo": model_name,
                "feature": feature_name,
                "coeficiente": coefficient,
                "nota": "Coeficiente em features escaladas. Valor zero sugere feature descartada pelo Lasso."
            })
        coefficient_rows.append({
            "modelo": model_name,
            "feature": "interceto",
            "coeficiente": model.intercept_,
            "nota": "Interceto do modelo Lasso."
        })

# -----------------------------
# Guarda dos resumos
# -----------------------------
pd.DataFrame(training_rows).to_csv(TAB_DIR / "03_tempos_treino.csv", index=False)
all_predictions.to_csv(PRED_DIR / "previsoes_todos_modelos.csv", index=False)

if coefficient_rows:
    coefficient_df = pd.DataFrame(coefficient_rows)
    coefficient_df.to_csv(TAB_DIR / "03_coeficientes_modelos_lineares.csv", index=False)
    coefficient_df[coefficient_df["modelo"] == "regressao_linear_multipla"].to_csv(
        TAB_DIR / "03_coeficientes_regressao_linear_multipla.csv", index=False
    )
    coefficient_df[coefficient_df["modelo"] == "lasso"].to_csv(
        TAB_DIR / "03_coeficientes_lasso.csv", index=False
    )

notes = []
notes.append("# Notas do Treino de Modelos\n")
notes.append("- Foram treinados cinco modelos: regressão linear simples, regressão linear múltipla, Ridge, Lasso e regressão polinomial de grau 2.\n")
notes.append("- O modelo simples usa apenas `distancia_planeada` para servir como baseline.\n")
notes.append("- Ridge e Lasso usam dados escalados, porque a regularização depende da escala dos coeficientes.\n")
notes.append("- O modelo polinomial também usa dados escalados para melhorar estabilidade numérica.\n")
notes.append("- As métricas não são calculadas neste script para manter responsabilidades separadas.\n")
notes.append("- As previsões e os modelos `.pkl` foram guardados para avaliação e reutilização.\n")

(NOTE_DIR / "03_notas_treino.md").write_text("".join(notes), encoding="utf-8")

print("\nFicheiros gerados:")
print(f"- Modelos: {MODEL_DIR}")
print(f"- Previsões: {PRED_DIR}")
print(f"- Tempos de treino: {TAB_DIR / '03_tempos_treino.csv'}")
print(f"- Coeficientes: {TAB_DIR / '03_coeficientes_modelos_lineares.csv'}")
print("\n03 - Treino de modelos concluído com sucesso.")
