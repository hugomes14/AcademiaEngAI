# 03_metodo_elbow.py
# Lab 07.1 — Clustering: Descoberta de Perfis Operacionais de Voo
# Etapa 3: Método Elbow e análise de k para K-Means
#
# Objetivo pedagógico:
# - Testar vários valores de k.
# - Calcular Inércia/WCSS, Silhouette, Davies-Bouldin e Calinski-Harabasz.
# - Guardar gráficos para apoiar a escolha do número de clusters.
# - Escolher automaticamente um k recomendado com base na Silhouette.

from pathlib import Path
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR = BASE_DIR / "outputs" / "figures" / "elbow"
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 07.1 — ETAPA 3: MÉTODO ELBOW")
print("=" * 80)

scaled_path = TABLE_DIR / "dados_escalonados.csv"
if not scaled_path.exists():
    raise FileNotFoundError("Ficheiro de dados escalonados não encontrado. Corre primeiro 02_preprocessamento.py.")

X = pd.read_csv(scaled_path)
print(f"\n[INFO] Dados escalonados carregados: {X.shape[0]} linhas x {X.shape[1]} features")

max_k = min(10, len(X) - 1)
k_values = list(range(2, max_k + 1))

resultados = []

for k in k_values:
    print(f"[INFO] A treinar K-Means com k={k}...")
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm="lloyd", max_iter=300)
    labels = modelo.fit_predict(X)

    inertia = float(modelo.inertia_)
    silhouette = float(silhouette_score(X, labels))
    dbi = float(davies_bouldin_score(X, labels))
    chi = float(calinski_harabasz_score(X, labels))

    resultados.append({
        "k": k,
        "inercia_wcss": inertia,
        "silhouette": silhouette,
        "davies_bouldin": dbi,
        "calinski_harabasz": chi
    })

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(TABLE_DIR / "elbow_metricas_kmeans.csv", index=False)

with open(TABLE_DIR / "elbow_metricas_kmeans.md", "w", encoding="utf-8") as f:
    f.write(df_resultados.to_markdown(index=False, floatfmt=".4f"))

print("\n[INFO] Resultados por k:")
print(df_resultados)

# Escolha automática: melhor Silhouette.
# Nota: o "cotovelo" deve ser analisado visualmente, mas a Silhouette dá uma regra objetiva.
melhor_linha = df_resultados.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
k_recomendado = int(melhor_linha["k"])

print(f"\n[OK] k recomendado automaticamente pela melhor Silhouette: {k_recomendado}")

with open(MODEL_DIR / "k_recomendado.json", "w", encoding="utf-8") as f:
    json.dump({
        "k_recomendado": k_recomendado,
        "criterio_automatico": "maior_silhouette",
        "nota": "O Método Elbow deve também ser avaliado visualmente pelo formando.",
        "melhor_silhouette": float(melhor_linha["silhouette"]),
        "davies_bouldin_no_k": float(melhor_linha["davies_bouldin"]),
        "calinski_harabasz_no_k": float(melhor_linha["calinski_harabasz"])
    }, f, indent=2, ensure_ascii=False)

# Gráfico do Método Elbow.
plt.figure(figsize=(9, 5))
plt.plot(df_resultados["k"], df_resultados["inercia_wcss"], marker="o")
plt.title("Método Elbow — Inércia/WCSS por número de clusters")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inércia / WCSS")
plt.xticks(k_values)
plt.tight_layout()
plt.savefig(FIG_DIR / "metodo_elbow_inercia.png", dpi=170)
plt.savefig(FIG_DIR / "metodo_elbow_inercia.pdf", dpi=170)
plt.close()

# Gráfico da Silhouette.
plt.figure(figsize=(9, 5))
plt.plot(df_resultados["k"], df_resultados["silhouette"], marker="o")
plt.axvline(k_recomendado, linestyle="--", label=f"k recomendado = {k_recomendado}")
plt.title("Coeficiente de Silhueta por k")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(k_values)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "silhouette_por_k.png", dpi=170)
plt.savefig(FIG_DIR / "silhouette_por_k.pdf", dpi=170)
plt.close()

# Gráfico com DBI e CHI em separado para evitar escalas confusas.
plt.figure(figsize=(9, 5))
plt.plot(df_resultados["k"], df_resultados["davies_bouldin"], marker="o")
plt.title("Índice Davies-Bouldin por k — menor é melhor")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(k_values)
plt.tight_layout()
plt.savefig(FIG_DIR / "davies_bouldin_por_k.png", dpi=170)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(df_resultados["k"], df_resultados["calinski_harabasz"], marker="o")
plt.title("Índice Calinski-Harabasz por k — maior é melhor")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Calinski-Harabasz Index")
plt.xticks(k_values)
plt.tight_layout()
plt.savefig(FIG_DIR / "calinski_harabasz_por_k.png", dpi=170)
plt.close()

with open(REPORT_DIR / "NOTAS_ELBOW.md", "w", encoding="utf-8") as f:
    f.write(f"""# Notas do Método Elbow — Lab 07.1

## k recomendado

- k recomendado automaticamente: **{k_recomendado}**
- Critério usado: maior Silhouette Score.

## Como interpretar

- A Inércia/WCSS desce sempre quando k aumenta. O objetivo é procurar o ponto em que a melhoria deixa de ser expressiva.
- A Silhouette varia entre -1 e 1. Valores mais altos indicam clusters mais compactos e separados.
- O Índice Davies-Bouldin deve ser baixo.
- O Índice Calinski-Harabasz deve ser alto.

## Nota importante

A escolha de k não é totalmente automática. Em contexto real, a interpretação operacional dos clusters também deve pesar na decisão.
""")

print("\n[OK] Método Elbow concluído.")
print(f"[OK] Tabela: {TABLE_DIR / 'elbow_metricas_kmeans.csv'}")
print(f"[OK] Gráficos: {FIG_DIR}")
print(f"[OK] k recomendado: {MODEL_DIR / 'k_recomendado.json'}")
