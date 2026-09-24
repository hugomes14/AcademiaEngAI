# 06_explicabilidade_pdp.py
# Lab 06.4 — Modelos Ensemble
# Etapa: Explicabilidade
#
# Esta etapa gera:
# - Visualização simplificada da Árvore de Decisão.
# - Importância de variáveis para Random Forest.
# - Importância de variáveis para XGBoost ou alternativa compatível.
# - Partial Dependence Plots para as variáveis mais relevantes.
#
# Nota pedagógica:
# A Feature Importance mostra quais variáveis mais influenciaram o modelo.
# O PDP mostra como a previsão muda quando uma variável varia, mantendo as outras
# constantes em média.

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import plot_tree

BASE_DIR = Path(__file__).resolve().parents[1]
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
METRIC_DIR = BASE_DIR / "artefactos" / "metricas"
FIG_DIR = BASE_DIR / "artefactos" / "figuras"

FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.4 — ETAPA 6: EXPLICABILIDADE")
print("=" * 80)

dados = joblib.load(PREP_DIR / "dados_processados.pkl")
X_train = dados["X_train"]
X_test = dados["X_test"]
y_train = dados["y_train"]

metadata = json.loads((PREP_DIR / "metadata_preprocessamento.json").read_text(encoding="utf-8"))
feature_names = metadata["feature_names"]

melhor = json.loads((METRIC_DIR / "melhor_modelo.json").read_text(encoding="utf-8"))
melhor_nome = melhor["modelo"]

modelos = {}
for ficheiro in MODEL_DIR.glob("*.pkl"):
    if ficheiro.name != "previsoes_teste.pkl":
        modelos[ficheiro.stem] = joblib.load(ficheiro)

print("\nModelos disponíveis para explicabilidade:")
for nome in modelos:
    print(f"- {nome}")

if "decision_tree" in modelos:
    arvore = modelos["decision_tree"]

    plt.figure(figsize=(22, 10))
    plot_tree(
        arvore,
        feature_names=feature_names,
        class_names=["sem_incidente", "incidente"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Visualização simplificada da Árvore de Decisão — primeiros níveis")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "arvore_decisao_primeiros_niveis.png", dpi=220)
    plt.savefig(FIG_DIR / "arvore_decisao_primeiros_niveis.pdf")
    plt.close()

    info_arvore = f"""# Informação da Árvore de Decisão

- Profundidade total: {arvore.get_depth()}
- Número de folhas: {arvore.get_n_leaves()}

Uma árvore sem limite de profundidade pode ficar demasiado ajustada ao treino.
Esta é uma boa demonstração de overfitting potencial.
"""
    (METRIC_DIR / "info_arvore_decisao.md").write_text(info_arvore, encoding="utf-8")

importancias_todas = []

for nome, modelo in modelos.items():
    if hasattr(modelo, "feature_importances_"):
        importancias = pd.DataFrame({
            "modelo": nome,
            "feature": feature_names,
            "importancia": modelo.feature_importances_,
        }).sort_values("importancia", ascending=False)

        importancias_todas.append(importancias)
        importancias.to_csv(METRIC_DIR / f"importancia_variaveis_{nome}.csv", index=False)
        importancias.round(4).to_markdown(METRIC_DIR / f"importancia_variaveis_{nome}.md", index=False)

        plt.figure(figsize=(9, 6))
        top = importancias.head(10).sort_values("importancia", ascending=True)
        plt.barh(top["feature"], top["importancia"])
        plt.title(f"Top variáveis mais importantes — {nome}")
        plt.xlabel("Importância")
        plt.ylabel("Variável")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"importancia_variaveis_{nome}.png", dpi=220)
        plt.savefig(FIG_DIR / f"importancia_variaveis_{nome}.pdf")
        plt.close()

if importancias_todas:
    todas = pd.concat(importancias_todas, ignore_index=True)
    todas.to_csv(METRIC_DIR / "importancia_variaveis_todos_modelos.csv", index=False)

    if "random_forest" in todas["modelo"].unique():
        base_importancias = todas[todas["modelo"] == "random_forest"].copy()
    else:
        base_importancias = todas[todas["modelo"] == melhor_nome].copy()

    top_features = base_importancias.sort_values("importancia", ascending=False)["feature"].head(2).tolist()
else:
    top_features = feature_names[:2]

print("\nVariáveis selecionadas para PDP:")
for feature in top_features:
    print(f"- {feature}")

modelo_pdp = modelos.get(melhor_nome)
if modelo_pdp is None:
    modelo_pdp = modelos.get("random_forest")

try:
    fig, ax = plt.subplots(figsize=(10, 5))
    PartialDependenceDisplay.from_estimator(
        modelo_pdp,
        X_test,
        features=top_features,
        ax=ax,
        kind="average",
    )
    plt.suptitle(f"Partial Dependence Plots — {melhor_nome}", y=1.05)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pdp_variaveis_importantes.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG_DIR / "pdp_variaveis_importantes.pdf", bbox_inches="tight")
    plt.close()
except Exception as erro:
    texto = f"Não foi possível gerar PDP automaticamente. Motivo: {erro}"
    print(texto)
    (METRIC_DIR / "erro_pdp.md").write_text(texto, encoding="utf-8")

resumo = f"""# Resumo de Explicabilidade

## Melhor modelo avaliado

- Modelo: `{melhor_nome}`

## Variáveis analisadas por PDP

{chr(10).join([f"- `{f}`" for f in top_features])}

## Leitura recomendada

- A importância de variáveis ajuda a identificar quais atributos pesam mais na decisão.
- O PDP ajuda a perceber se o aumento de uma variável tende a aumentar ou reduzir a probabilidade prevista de incidente.
- Como o dataset é desbalanceado, a interpretação deve ser feita em conjunto com métricas como Recall, F1 e PR-AUC.
"""
(METRIC_DIR / "resumo_explicabilidade.md").write_text(resumo, encoding="utf-8")

print("\nArtefactos de explicabilidade guardados em:")
print(FIG_DIR)
print(METRIC_DIR)
print("\nEtapa 6 concluída com sucesso.")
