# 06_dbscan_perfil_clusters.py
# Lab 07.1 — Clustering: Descoberta de Perfis Operacionais de Voo
# Etapa 6: Visualização de DBSCAN e perfil dos clusters
#
# Objetivo pedagógico:
# - Visualizar os clusters do DBSCAN com PCA e destacar ruído.
# - Criar perfis dos clusters do melhor modelo usando os dados originais.
# - Criar descrições operacionais simples para cada cluster.

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
DATASET_PATH = BASE_DIR / "data" / "voos_telemetria_completa.csv"
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR = BASE_DIR / "outputs" / "figures" / "perfil_clusters"
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 07.1 — ETAPA 6: DBSCAN E PERFIL DOS CLUSTERS")
print("=" * 80)

df_original = pd.read_csv(DATASET_PATH)
labels_df = pd.read_csv(TABLE_DIR / "labels_modelos.csv")
pca_df = pd.read_csv(TABLE_DIR / "componentes_pca_clusters.csv")

melhor_modelo_path = MODEL_DIR / "melhor_modelo_clustering.txt"
if melhor_modelo_path.exists():
    melhor_modelo = melhor_modelo_path.read_text(encoding="utf-8").strip()
else:
    melhor_modelo = "kmeans"

print(f"\n[INFO] Melhor modelo usado para profiling: {melhor_modelo}")

# Visualização DBSCAN com destaque para ruído.
if "dbscan" in labels_df.columns:
    labels_dbscan = labels_df["dbscan"].astype(int)
    plot_df = pca_df.copy()
    plot_df["label_dbscan"] = labels_dbscan.astype(str)
    plot_df["tipo"] = np.where(labels_dbscan == -1, "Ruído / outlier", "Cluster")

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="label_dbscan",
        style="tipo",
        palette="tab10",
        s=80,
        edgecolor="black",
        linewidth=0.3
    )
    plt.title("Visualização PCA — DBSCAN com Ruído Destacado")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Label DBSCAN", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_dbscan_ruido_destacado.png", dpi=180)
    plt.savefig(FIG_DIR / "pca_dbscan_ruido_destacado.pdf", dpi=180)
    plt.close()

    n_ruido = int((labels_dbscan == -1).sum())
    print(f"[INFO] Observações marcadas como ruído pelo DBSCAN: {n_ruido}")
else:
    n_ruido = 0
    print("[AVISO] Labels do DBSCAN não encontrados.")

if melhor_modelo not in labels_df.columns:
    raise ValueError(f"Labels do melhor modelo '{melhor_modelo}' não existem em labels_modelos.csv")

df_perfil = df_original.copy()
df_perfil["cluster"] = labels_df[melhor_modelo].astype(int)

numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()

perfil_media = df_perfil.groupby("cluster")[numeric_cols].mean().round(3)
perfil_mediana = df_perfil.groupby("cluster")[numeric_cols].median().round(3)
perfil_count = df_perfil.groupby("cluster").size().rename("n_voos").to_frame()

perfil_completo = perfil_count.join(perfil_media, how="left")
perfil_completo.to_csv(TABLE_DIR / "perfil_clusters_medias.csv")

with open(TABLE_DIR / "perfil_clusters_medias.md", "w", encoding="utf-8") as f:
    f.write(perfil_completo.to_markdown(floatfmt=".3f"))

perfil_mediana.to_csv(TABLE_DIR / "perfil_clusters_medianas.csv")

df_perfil.to_csv(TABLE_DIR / "dados_originais_com_clusters.csv", index=False)

print("\n[INFO] Perfil médio por cluster:")
print(perfil_completo)

# Gráfico de médias por cluster. Para visualização comparável, faz-se normalização min-max apenas das médias.
perfil_plot = perfil_media.copy()
perfil_norm = (perfil_plot - perfil_plot.min()) / (perfil_plot.max() - perfil_plot.min())
perfil_norm = perfil_norm.fillna(0)
perfil_norm["cluster"] = perfil_norm.index.astype(str)

perfil_long = perfil_norm.melt(id_vars="cluster", var_name="feature", value_name="valor_normalizado")

plt.figure(figsize=(13, 7))
sns.barplot(data=perfil_long, x="feature", y="valor_normalizado", hue="cluster")
plt.title("Perfil dos Clusters — Médias Normalizadas por Feature")
plt.xlabel("Feature")
plt.ylabel("Média normalizada entre clusters")
plt.xticks(rotation=35, ha="right")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(FIG_DIR / "perfil_clusters_medias_normalizadas.png", dpi=180)
plt.savefig(FIG_DIR / "perfil_clusters_medias_normalizadas.pdf", dpi=180)
plt.close()

# Heatmap do perfil.
plt.figure(figsize=(11, 6))
sns.heatmap(perfil_norm.drop(columns=["cluster"]), annot=True, fmt=".2f", cmap="viridis")
plt.title("Heatmap do Perfil dos Clusters — Médias Normalizadas")
plt.xlabel("Features")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(FIG_DIR / "heatmap_perfil_clusters.png", dpi=180)
plt.close()

# Personas simples: compara cada cluster com a média global.
global_means = df_original[numeric_cols].mean()
personas = []

for cluster_id, row in perfil_media.iterrows():
    maiores = []
    menores = []

    for col in numeric_cols:
        if row[col] >= global_means[col] * 1.10:
            maiores.append(col)
        elif row[col] <= global_means[col] * 0.90:
            menores.append(col)

    # Nome simples com base nas variáveis mais destacadas.
    if "duracao_voo_min" in maiores and "distancia_percorrida_km" in maiores:
        nome = "Missões longas e extensas"
    elif "variacao_vertical_total_m" in maiores:
        nome = "Voos com elevada manobra vertical"
    elif "altitude_maxima_m" in maiores:
        nome = "Voos de altitude elevada"
    elif "duracao_voo_min" in menores and "distancia_percorrida_km" in menores:
        nome = "Voos curtos e compactos"
    else:
        nome = "Perfil operacional intermédio"

    personas.append({
        "cluster": int(cluster_id),
        "nome_sugerido": nome,
        "n_voos": int(perfil_count.loc[cluster_id, "n_voos"]),
        "features_acima_da_media": ", ".join(maiores) if maiores else "nenhuma",
        "features_abaixo_da_media": ", ".join(menores) if menores else "nenhuma",
        "interpretacao": (
            f"O cluster {cluster_id} foi caracterizado como '{nome}'. "
            f"As variáveis acima da média global são: {', '.join(maiores) if maiores else 'nenhuma'}. "
            f"As variáveis abaixo da média global são: {', '.join(menores) if menores else 'nenhuma'}."
        )
    })

df_personas = pd.DataFrame(personas)
df_personas.to_csv(TABLE_DIR / "personas_clusters.csv", index=False)

with open(TABLE_DIR / "personas_clusters.md", "w", encoding="utf-8") as f:
    f.write(df_personas.to_markdown(index=False))

with open(REPORT_DIR / "NOTAS_PERFIL_CLUSTERS.md", "w", encoding="utf-8") as f:
    f.write("# Notas de Perfil dos Clusters — Lab 07.1\n\n")
    f.write(f"Modelo usado para profiling: **{melhor_modelo}**\n\n")
    f.write("## Personas sugeridas\n\n")
    f.write(df_personas.to_markdown(index=False))
    f.write("\n\n## Nota\n\n")
    f.write("As personas são descrições pedagógicas. Devem ser revistas por alguém com conhecimento do contexto operacional dos voos.\n")

print("\n[INFO] Personas sugeridas:")
print(df_personas[["cluster", "nome_sugerido", "n_voos"]])

print("\n[OK] Perfil dos clusters concluído.")
print(f"[OK] Tabela de perfil: {TABLE_DIR / 'perfil_clusters_medias.csv'}")
print(f"[OK] Personas: {TABLE_DIR / 'personas_clusters.csv'}")
print(f"[OK] Figuras: {FIG_DIR}")
