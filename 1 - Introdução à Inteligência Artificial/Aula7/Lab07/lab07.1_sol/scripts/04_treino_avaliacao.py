# 04_treino_avaliacao.py
# Lab 07.1 — Clustering: Descoberta de Perfis Operacionais de Voo
# Etapa 4: Treino de modelos e avaliação
#
# Objetivo pedagógico:
# - Treinar K-Means, DBSCAN e Agglomerative Clustering.
# - Avaliar clusters com métricas internas.
# - Guardar modelos, labels e tabelas comparativas.
# - Criar dendrograma para ajudar a interpretar a estrutura hierárquica.

from pathlib import Path
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR = BASE_DIR / "outputs" / "figures" / "modelos"
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 07.1 — ETAPA 4: TREINO E AVALIAÇÃO")
print("=" * 80)

scaled_path = TABLE_DIR / "dados_escalonados.csv"
if not scaled_path.exists():
    raise FileNotFoundError("Ficheiro de dados escalonados não encontrado. Corre primeiro 02_preprocessamento.py.")

X = pd.read_csv(scaled_path)

k_json = MODEL_DIR / "k_recomendado.json"
if k_json.exists():
    with open(k_json, "r", encoding="utf-8") as f:
        k_info = json.load(f)
    k_otimo = int(k_info["k_recomendado"])
else:
    k_otimo = 3
    print("[AVISO] k_recomendado.json não encontrado. Será usado k=3.")

print(f"\n[INFO] k usado para K-Means e AgglomerativeClustering: {k_otimo}")

# Função lógica inline para calcular métricas com segurança.
# Nota: está escrita de forma explícita para ser fácil de ler.
resultados = []
labels_dict = {}

# K-Means.
print("\n[INFO] Treino do K-Means...")
kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10, algorithm="lloyd", max_iter=300)
labels_kmeans = kmeans.fit_predict(X)
labels_dict["kmeans"] = labels_kmeans
joblib.dump(kmeans, MODEL_DIR / "modelo_kmeans.pkl")

# Agglomerative Clustering.
print("[INFO] Treino do AgglomerativeClustering...")
agg = AgglomerativeClustering(n_clusters=k_otimo, linkage="ward")
labels_agg = agg.fit_predict(X)
labels_dict["agglomerative"] = labels_agg
joblib.dump({"modelo": agg, "labels": labels_agg}, MODEL_DIR / "modelo_agglomerative.pkl")

# DBSCAN.
# eps e min_samples controlam a densidade necessária para formar clusters.
# min_samples costuma começar em n_features + 1.
print("[INFO] Procura automática simples de parâmetros para DBSCAN...")
min_samples = max(4, X.shape[1] + 1)

# Estimativa inicial por distâncias aos k-vizinhos.
nn = NearestNeighbors(n_neighbors=min_samples)
nn.fit(X)
distancias, _ = nn.kneighbors(X)
k_distancias = np.sort(distancias[:, -1])

plt.figure(figsize=(9, 5))
plt.plot(k_distancias)
plt.title(f"Gráfico k-distância para DBSCAN — min_samples={min_samples}")
plt.xlabel("Observações ordenadas")
plt.ylabel(f"Distância ao {min_samples}.º vizinho")
plt.tight_layout()
plt.savefig(FIG_DIR / "dbscan_k_distancia.png", dpi=170)
plt.close()

eps_candidatos = np.quantile(k_distancias, [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]).tolist()
melhor_dbscan = None
melhor_dbscan_score = -999

for eps in eps_candidatos:
    modelo_tmp = DBSCAN(eps=float(eps), min_samples=min_samples)
    labels_tmp = modelo_tmp.fit_predict(X)
    labels_unicos = sorted(set(labels_tmp.tolist()))
    n_clusters_sem_ruido = len([l for l in labels_unicos if l != -1])
    n_noise = int((labels_tmp == -1).sum())

    if len(labels_unicos) > 1 and n_clusters_sem_ruido >= 1:
        try:
            score = float(silhouette_score(X, labels_tmp))
        except Exception:
            score = -999
    else:
        score = -999

    print(f"  eps={eps:.4f} | clusters_sem_ruido={n_clusters_sem_ruido} | ruído={n_noise} | silhouette={score:.4f}")

    if score > melhor_dbscan_score:
        melhor_dbscan_score = score
        melhor_dbscan = (float(eps), labels_tmp)

if melhor_dbscan is None:
    eps_final = float(np.quantile(k_distancias, 0.90))
    labels_dbscan = DBSCAN(eps=eps_final, min_samples=min_samples).fit_predict(X)
else:
    eps_final, labels_dbscan = melhor_dbscan

dbscan = DBSCAN(eps=eps_final, min_samples=min_samples)
labels_dbscan = dbscan.fit_predict(X)
labels_dict["dbscan"] = labels_dbscan
joblib.dump(dbscan, MODEL_DIR / "modelo_dbscan.pkl")

print(f"[OK] DBSCAN final: eps={eps_final:.4f}, min_samples={min_samples}")

# Avaliação dos modelos.
for nome_modelo, labels in labels_dict.items():
    labels_unicos = sorted(set(labels.tolist()))
    n_labels = len(labels_unicos)
    n_clusters_sem_ruido = len([l for l in labels_unicos if l != -1])
    n_ruido = int((labels == -1).sum())

    if n_labels > 1:
        silhouette = float(silhouette_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))
        chi = float(calinski_harabasz_score(X, labels))
    else:
        silhouette = np.nan
        dbi = np.nan
        chi = np.nan

    resultados.append({
        "modelo": nome_modelo,
        "parametros": (
            f"k={k_otimo}" if nome_modelo in ["kmeans", "agglomerative"]
            else f"eps={eps_final:.4f}; min_samples={min_samples}"
        ),
        "n_labels_total": n_labels,
        "n_clusters_sem_ruido": n_clusters_sem_ruido,
        "n_ruido": n_ruido,
        "silhouette": silhouette,
        "davies_bouldin": dbi,
        "calinski_harabasz": chi
    })

df_metricas = pd.DataFrame(resultados)
df_metricas = df_metricas.sort_values("silhouette", ascending=False, na_position="last").reset_index(drop=True)

df_metricas.to_csv(TABLE_DIR / "metricas_clustering.csv", index=False)
with open(TABLE_DIR / "metricas_clustering.md", "w", encoding="utf-8") as f:
    f.write(df_metricas.to_markdown(index=False, floatfmt=".4f"))

labels_df = pd.DataFrame(labels_dict)
labels_df.to_csv(TABLE_DIR / "labels_modelos.csv", index=False)

with open(MODEL_DIR / "parametros_dbscan.json", "w", encoding="utf-8") as f:
    json.dump({"eps": eps_final, "min_samples": min_samples}, f, indent=2, ensure_ascii=False)

print("\n[INFO] Métricas de clustering:")
print(df_metricas)

# Gráfico comparativo das métricas principais.
metricas_plot = df_metricas.melt(
    id_vars=["modelo"],
    value_vars=["silhouette", "davies_bouldin", "calinski_harabasz"],
    var_name="metrica",
    value_name="valor"
)

for metrica in ["silhouette", "davies_bouldin", "calinski_harabasz"]:
    plt.figure(figsize=(8, 5))
    plt.bar(df_metricas["modelo"].astype(str), df_metricas[metrica])
    plt.title(f"Comparação de modelos — {metrica}")
    plt.xlabel("Modelo")
    plt.ylabel(metrica)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"comparacao_{metrica}.png", dpi=170)
    plt.close()

# Dendrograma para interpretação hierárquica.
try:
    from scipy.cluster.hierarchy import linkage, dendrogram

    print("\n[INFO] A gerar dendrograma...")
    X_np = X.to_numpy()
    if len(X_np) > 120:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_np), size=120, replace=False)
        X_dendro = X_np[idx]
        titulo_extra = "amostra de 120 observações"
    else:
        X_dendro = X_np
        titulo_extra = "dataset completo"

    Z = linkage(X_dendro, method="ward")
    plt.figure(figsize=(14, 7))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=10)
    plt.title(f"Dendrograma — Clustering Hierárquico ({titulo_extra})")
    plt.xlabel("Observações ou grupos agregados")
    plt.ylabel("Distância")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dendrograma_hierarquico.png", dpi=170)
    plt.savefig(FIG_DIR / "dendrograma_hierarquico.pdf", dpi=170)
    plt.close()
except Exception as exc:
    print(f"[AVISO] Dendrograma não gerado: {exc}")

with open(REPORT_DIR / "NOTAS_MODELOS.md", "w", encoding="utf-8") as f:
    f.write("""# Notas de Treino e Avaliação — Lab 07.1

## Modelos treinados

- K-Means
- DBSCAN
- AgglomerativeClustering

## Métricas internas

- Silhouette Score: quanto mais próximo de 1, melhor.
- Davies-Bouldin Index: quanto mais próximo de 0, melhor.
- Calinski-Harabasz Index: quanto maior, melhor.

## DBSCAN

O DBSCAN identifica regiões densas e pode marcar observações como ruído (`label = -1`). Isto é útil quando há outliers ou grupos de forma irregular.

## Interpretação

As métricas ajudam, mas a escolha final deve também considerar se os clusters fazem sentido no contexto operacional.
""")

print("\n[OK] Treino e avaliação concluídos.")
print(f"[OK] Métricas: {TABLE_DIR / 'metricas_clustering.csv'}")
print(f"[OK] Labels: {TABLE_DIR / 'labels_modelos.csv'}")
print(f"[OK] Modelos: {MODEL_DIR}")
