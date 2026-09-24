# 03_pca_selecao_componentes.py
# Lab 07.2 — Redução de Dimensionalidade
# Etapa 3: PCA completo, Scree Plot e seleção do número de componentes.
#
# Nota pedagógica:
# - Este ficheiro corre de forma direta, sem funções e sem main.
# - O PCA completo calcula a variância explicada por todos os componentes.
# - A escolha de k pode seguir 95% de variância acumulada ou o cotovelo visual.

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
MODELOS_DIR = BASE_DIR / "modelos"
TABELAS_DIR = BASE_DIR / "tabelas"
IMAGENS_DIR = BASE_DIR / "imagens"

for directory in [ARTEFACTOS_DIR, MODELOS_DIR, TABELAS_DIR, IMAGENS_DIR]:
    directory.mkdir(exist_ok=True)

print("=" * 80)
print("LAB 07.2 — ETAPA 3: PCA E SELEÇÃO DE COMPONENTES")
print("=" * 80)

X_scaled_path = ARTEFACTOS_DIR / "X_scaled.csv"
if not X_scaled_path.exists():
    raise FileNotFoundError("Execute primeiro 02_preprocessamento.py para criar artefactos/X_scaled.csv")

X_scaled = pd.read_csv(X_scaled_path)
print(f"X_scaled carregado com dimensão: {X_scaled.shape}")

pca_full = PCA(n_components=None, random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)

explained = pca_full.explained_variance_ratio_
cum_explained = np.cumsum(explained)
components = np.arange(1, len(explained) + 1)

k_90 = int(np.argmax(cum_explained >= 0.90) + 1)
k_95 = int(np.argmax(cum_explained >= 0.95) + 1)
k_99 = int(np.argmax(cum_explained >= 0.99) + 1)

print("\nComponentes necessários:")
print(f"- 90% da variância: {k_90}")
print(f"- 95% da variância: {k_95}")
print(f"- 99% da variância: {k_99}")

variance_df = pd.DataFrame({
    "componente": [f"PC{i}" for i in components],
    "numero_componente": components,
    "variancia_explicada": explained,
    "variancia_explicada_percentagem": explained * 100,
    "variancia_acumulada": cum_explained,
    "variancia_acumulada_percentagem": cum_explained * 100,
})

# Calcula erro de reconstrução por k, para quantificar a perda de informação.
erros = []
for k in components:
    pca_k = PCA(n_components=int(k), random_state=42)
    X_reduzido = pca_k.fit_transform(X_scaled)
    X_reconstruido = pca_k.inverse_transform(X_reduzido)
    mse = mean_squared_error(X_scaled, X_reconstruido)
    erros.append(mse)

variance_df["erro_reconstrucao_mse"] = erros

variance_df.to_csv(TABELAS_DIR / "03_pca_variancia_componentes.csv", index=False, encoding="utf-8")
variance_df.to_markdown(TABELAS_DIR / "03_pca_variancia_componentes.md", index=False)
joblib.dump(pca_full, MODELOS_DIR / "pca_full.pkl")

metadata = {
    "n_features": int(X_scaled.shape[1]),
    "k_90": k_90,
    "k_95": k_95,
    "k_99": k_99,
    "criterio_recomendado": "Usar k_95 para reter 95% da variância.",
}
(ARTEFACTOS_DIR / "pca_selecao_componentes_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# Scree Plot: variância explicada por componente.
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(components, explained * 100)
ax.plot(components, explained * 100, marker="o")
ax.set_title("Scree Plot — Variância explicada por componente")
ax.set_xlabel("Componente principal")
ax.set_ylabel("Variância explicada (%)")
ax.set_xticks(components)
ax.tick_params(axis="x", labelrotation=90)
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "03_scree_plot_variancia.png", dpi=180)
fig.savefig(IMAGENS_DIR / "03_scree_plot_variancia.pdf")
plt.close(fig)

# Variância acumulada.
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(components, cum_explained * 100, marker="o")
ax.axhline(90, linestyle="--", linewidth=1)
ax.axhline(95, linestyle="--", linewidth=1)
ax.axhline(99, linestyle="--", linewidth=1)
ax.axvline(k_95, linestyle="--", linewidth=1)
ax.set_title("Variância explicada acumulada")
ax.set_xlabel("Número de componentes")
ax.set_ylabel("Variância acumulada (%)")
ax.set_xticks(components)
ax.tick_params(axis="x", labelrotation=90)
ax.text(k_95, 96, f"k={k_95} para 95%", ha="left")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "03_variancia_acumulada.png", dpi=180)
fig.savefig(IMAGENS_DIR / "03_variancia_acumulada.pdf")
plt.close(fig)

# Erro de reconstrução por k.
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(components, erros, marker="o")
ax.axvline(k_95, linestyle="--", linewidth=1)
ax.set_title("Erro de reconstrução por número de componentes")
ax.set_xlabel("Número de componentes")
ax.set_ylabel("MSE entre X_scaled e X reconstruído")
ax.set_xticks(components)
ax.tick_params(axis="x", labelrotation=90)
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "03_erro_reconstrucao_por_k.png", dpi=180)
fig.savefig(IMAGENS_DIR / "03_erro_reconstrucao_por_k.pdf")
plt.close(fig)

print("\nTabelas e imagens guardadas:")
print("- tabelas/03_pca_variancia_componentes.csv")
print("- imagens/03_scree_plot_variancia.png")
print("- imagens/03_variancia_acumulada.png")
print("- imagens/03_erro_reconstrucao_por_k.png")
print("- modelos/pca_full.pkl")
print("\nEtapa 3 concluída com sucesso.")
