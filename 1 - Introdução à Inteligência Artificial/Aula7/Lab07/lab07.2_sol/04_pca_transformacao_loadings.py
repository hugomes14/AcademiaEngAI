# 04_pca_transformacao_loadings.py
# Lab 07.2 — Redução de Dimensionalidade
# Etapa 4: treino do PCA final, transformação e análise de loadings.
#
# Nota pedagógica:
# - Este ficheiro corre de forma direta, sem funções e sem main.
# - Os loadings ajudam a interpretar o peso das features originais nos componentes.
# - Quanto maior o valor absoluto do loading, maior o contributo dessa feature.

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
MODELOS_DIR = BASE_DIR / "modelos"
TABELAS_DIR = BASE_DIR / "tabelas"
IMAGENS_DIR = BASE_DIR / "imagens"

for directory in [ARTEFACTOS_DIR, MODELOS_DIR, TABELAS_DIR, IMAGENS_DIR]:
    directory.mkdir(exist_ok=True)

print("=" * 80)
print("LAB 07.2 — ETAPA 4: PCA FINAL E LOADINGS")
print("=" * 80)

X_scaled_path = ARTEFACTOS_DIR / "X_scaled.csv"
variance_path = TABELAS_DIR / "03_pca_variancia_componentes.csv"

if not X_scaled_path.exists():
    raise FileNotFoundError("Execute primeiro 02_preprocessamento.py.")
if not variance_path.exists():
    raise FileNotFoundError("Execute primeiro 03_pca_selecao_componentes.py.")

X_scaled = pd.read_csv(X_scaled_path)
variance_df = pd.read_csv(variance_path)
feature_names = X_scaled.columns.tolist()

k_95 = int(variance_df.loc[variance_df["variancia_acumulada"] >= 0.95, "numero_componente"].iloc[0])
print(f"Número de componentes escolhido para 95% da variância: {k_95}")

pca_final = PCA(n_components=k_95, random_state=42)
X_pca = pca_final.fit_transform(X_scaled)

pca_cols = [f"PC{i}" for i in range(1, k_95 + 1)]
X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
X_pca_df.to_csv(ARTEFACTOS_DIR / "X_pca.csv", index=False, encoding="utf-8")

joblib.dump(pca_final, MODELOS_DIR / "pca_model_95.pkl")

loadings = pd.DataFrame(
    pca_final.components_,
    index=pca_cols,
    columns=feature_names
)
loadings.to_csv(TABELAS_DIR / "04_pca_loadings.csv", encoding="utf-8")
loadings.to_markdown(TABELAS_DIR / "04_pca_loadings.md")

# Tabela com as features mais relevantes nos primeiros componentes.
top_rows = []
for pc in pca_cols[: min(10, len(pca_cols))]:
    valores = loadings.loc[pc].sort_values(key=lambda s: s.abs(), ascending=False).head(10)
    for feature, loading in valores.items():
        top_rows.append({
            "componente": pc,
            "feature": feature,
            "loading": loading,
            "abs_loading": abs(loading)
        })

top_loadings = pd.DataFrame(top_rows)
top_loadings.to_csv(TABELAS_DIR / "04_top_loadings_por_componente.csv", index=False, encoding="utf-8")
top_loadings.to_markdown(TABELAS_DIR / "04_top_loadings_por_componente.md", index=False)

# Heatmap legível: primeiros 10 componentes e 25 features mais relevantes no conjunto.
componentes_heatmap = pca_cols[: min(10, len(pca_cols))]
features_relevantes = (
    loadings.loc[componentes_heatmap]
    .abs()
    .sum(axis=0)
    .sort_values(ascending=False)
    .head(min(25, len(feature_names)))
    .index
    .tolist()
)

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(loadings.loc[componentes_heatmap, features_relevantes], cmap="coolwarm", center=0, ax=ax)
ax.set_title("Heatmap de loadings — componentes principais vs. sensores")
ax.set_xlabel("Features originais")
ax.set_ylabel("Componentes principais")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "04_heatmap_loadings_pca.png", dpi=180)
fig.savefig(IMAGENS_DIR / "04_heatmap_loadings_pca.pdf")
plt.close(fig)

# Gráfico de barras dos loadings absolutos de PC1 e PC2.
for pc in pca_cols[: min(2, len(pca_cols))]:
    valores = loadings.loc[pc].sort_values(key=lambda s: s.abs(), ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    valores.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top 15 loadings de {pc}")
    ax.set_xlabel("Loading")
    fig.tight_layout()
    fig.savefig(IMAGENS_DIR / f"04_top_loadings_{pc}.png", dpi=180)
    fig.savefig(IMAGENS_DIR / f"04_top_loadings_{pc}.pdf")
    plt.close(fig)

metadata = {
    "n_components_95": int(k_95),
    "variancia_total_retida": float(pca_final.explained_variance_ratio_.sum()),
    "pca_model": "modelos/pca_model_95.pkl",
    "X_pca": "artefactos/X_pca.csv"
}
(ARTEFACTOS_DIR / "pca_final_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nArtefactos guardados:")
print("- artefactos/X_pca.csv")
print("- modelos/pca_model_95.pkl")
print("- tabelas/04_pca_loadings.csv")
print("- tabelas/04_top_loadings_por_componente.csv")
print("- imagens/04_heatmap_loadings_pca.png")
print("\nInterpretação:")
print("- Loadings positivos e negativos indicam direção no componente.")
print("- O valor absoluto indica intensidade do contributo.")
print("- Sensores com maior loading absoluto em PC1/PC2 explicam grande parte da estrutura projetada.")
print("\nEtapa 4 concluída com sucesso.")
