# 05_visualizacao_pca.py
# Lab 07.1 — Clustering: Descoberta de Perfis Operacionais de Voo
# Etapa 5: Visualização de clusters com PCA
#
# Objetivo pedagógico:
# - Reduzir a dimensionalidade para 2 componentes.
# - Visualizar clusters em 2D.
# - Selecionar o melhor modelo por Silhouette.
# - Mostrar centróides quando o melhor modelo é K-Means.

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR = BASE_DIR / "outputs" / "figures" / "pca"
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 07.1 — ETAPA 5: VISUALIZAÇÃO PCA")
print("=" * 80)

X = pd.read_csv(TABLE_DIR / "dados_escalonados.csv")
labels_df = pd.read_csv(TABLE_DIR / "labels_modelos.csv")
metricas = pd.read_csv(TABLE_DIR / "metricas_clustering.csv")

validas = metricas.dropna(subset=["silhouette"]).copy()
if validas.empty:
    raise ValueError("Não há métricas válidas para selecionar o melhor modelo.")

melhor_modelo = validas.sort_values("silhouette", ascending=False).iloc[0]["modelo"]
print(f"\n[OK] Melhor modelo por Silhouette: {melhor_modelo}")

pca = PCA(n_components=2, random_state=42)
componentes = pca.fit_transform(X)

pca_df = pd.DataFrame({
    "PC1": componentes[:, 0],
    "PC2": componentes[:, 1]
})

for col in labels_df.columns:
    pca_df[f"cluster_{col}"] = labels_df[col].astype(str)

pca_df.to_csv(TABLE_DIR / "componentes_pca_clusters.csv", index=False)
joblib.dump(pca, MODEL_DIR / "pca_2_componentes.pkl")

variancia = pd.DataFrame({
    "componente": ["PC1", "PC2"],
    "variancia_explicada": pca.explained_variance_ratio_,
    "variancia_explicada_percent": pca.explained_variance_ratio_ * 100
})
variancia.to_csv(TABLE_DIR / "pca_variancia_explicada.csv", index=False)

print("\n[INFO] Variância explicada:")
print(variancia)

# Gráfico do melhor modelo.
labels_melhor = labels_df[melhor_modelo]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue=labels_melhor.astype(str),
    palette="tab10",
    s=70,
    edgecolor="black",
    linewidth=0.3
)

# Centróides do K-Means no espaço PCA.
if melhor_modelo == "kmeans":
    try:
        kmeans = joblib.load(MODEL_DIR / "modelo_kmeans.pkl")
        centroides_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centroides_pca[:, 0],
            centroides_pca[:, 1],
            s=260,
            marker="X",
            c="black",
            label="Centróides K-Means"
        )
    except Exception as exc:
        print(f"[AVISO] Não foi possível adicionar centróides: {exc}")

plt.title(f"Visualização PCA dos Clusters — Melhor Modelo: {melhor_modelo}")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variância)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variância)")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "pca_melhor_modelo.png", dpi=180)
plt.savefig(FIG_DIR / "pca_melhor_modelo.pdf", dpi=180)
plt.close()

# Gráficos de todos os modelos para comparação visual.
for modelo in labels_df.columns:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=labels_df[modelo].astype(str),
        palette="tab10",
        s=70,
        edgecolor="black",
        linewidth=0.3
    )
    plt.title(f"Visualização PCA — {modelo}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variância)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variância)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"pca_{modelo}.png", dpi=180)
    plt.close()

with open(MODEL_DIR / "melhor_modelo_clustering.txt", "w", encoding="utf-8") as f:
    f.write(str(melhor_modelo))

with open(REPORT_DIR / "NOTAS_PCA.md", "w", encoding="utf-8") as f:
    f.write(f"""# Notas da Visualização PCA — Lab 07.1

## Melhor modelo

- Melhor modelo por Silhouette: **{melhor_modelo}**

## Variância explicada

- PC1: {pca.explained_variance_ratio_[0] * 100:.2f}%
- PC2: {pca.explained_variance_ratio_[1] * 100:.2f}%
- Total nos dois componentes: {pca.explained_variance_ratio_.sum() * 100:.2f}%

## Interpretação

A PCA reduz as features para duas dimensões apenas para visualização. A separação visual ajuda a interpretar os grupos, mas as métricas foram calculadas no espaço completo escalonado.
""")

print("\n[OK] Visualização PCA concluída.")
print(f"[OK] Melhor modelo: {melhor_modelo}")
print(f"[OK] Figuras: {FIG_DIR}")
