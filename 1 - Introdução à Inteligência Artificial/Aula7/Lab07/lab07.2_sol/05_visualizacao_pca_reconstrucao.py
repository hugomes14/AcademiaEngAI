# 05_visualizacao_pca_reconstrucao.py
# Lab 07.2 — Redução de Dimensionalidade
# Etapa 5: visualização PCA 2D e cálculo do erro de reconstrução.
#
# Nota pedagógica:
# - Este ficheiro corre de forma direta, sem funções e sem main.
# - A visualização 2D usa PC1 e PC2, mas o modelo PCA final pode ter mais componentes.
# - O erro de reconstrução mede a perda de informação causada pela redução.

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
print("LAB 07.2 — ETAPA 5: VISUALIZAÇÃO PCA E RECONSTRUÇÃO")
print("=" * 80)

X_pca_path = ARTEFACTOS_DIR / "X_pca.csv"
X_scaled_path = ARTEFACTOS_DIR / "X_scaled.csv"
y_path = ARTEFACTOS_DIR / "y.csv"
pca_model_path = MODELOS_DIR / "pca_model_95.pkl"

for path in [X_pca_path, X_scaled_path, y_path, pca_model_path]:
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro obrigatório em falta: {path}")

X_pca = pd.read_csv(X_pca_path)
X_scaled = pd.read_csv(X_scaled_path)
y = pd.read_csv(y_path).iloc[:, 0]
pca_model = joblib.load(pca_model_path)

print(f"X_pca carregado: {X_pca.shape}")
print(f"X_scaled carregado: {X_scaled.shape}")
print(f"Classes/cores disponíveis: {sorted(y.astype(str).unique())}")

# Gráfico PCA 2D.
plot_df = X_pca[["PC1", "PC2"]].copy()
plot_df["tipo_manobra"] = y.astype(str)

fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="tipo_manobra", alpha=0.8, ax=ax)
ax.set_title("Projeção PCA 2D — PC1 vs. PC2")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(title="Tipo de manobra", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "05_pca_2d_tipo_manobra.png", dpi=180)
fig.savefig(IMAGENS_DIR / "05_pca_2d_tipo_manobra.pdf")
plt.close(fig)

# Erro de reconstrução.
X_reconstruido = pca_model.inverse_transform(X_pca)
mse_reconstrucao = mean_squared_error(X_scaled, X_reconstruido)

metricas = pd.DataFrame([{
    "modelo": "PCA_95_variancia",
    "n_componentes": int(pca_model.n_components_),
    "variancia_retida": float(pca_model.explained_variance_ratio_.sum()),
    "erro_reconstrucao_mse": float(mse_reconstrucao)
}])

metricas.to_csv(TABELAS_DIR / "05_metricas_reconstrucao_pca.csv", index=False, encoding="utf-8")
metricas.to_markdown(TABELAS_DIR / "05_metricas_reconstrucao_pca.md", index=False)

# Visualização do erro médio absoluto por feature reconstruída.
erro_por_feature = (pd.DataFrame(X_reconstruido, columns=X_scaled.columns) - X_scaled).abs().mean().sort_values(ascending=False)
erro_por_feature_df = erro_por_feature.reset_index()
erro_por_feature_df.columns = ["feature", "erro_absoluto_medio_reconstrucao"]
erro_por_feature_df.to_csv(TABELAS_DIR / "05_erro_reconstrucao_por_feature.csv", index=False, encoding="utf-8")
erro_por_feature_df.to_markdown(TABELAS_DIR / "05_erro_reconstrucao_por_feature.md", index=False)

fig, ax = plt.subplots(figsize=(12, 7))
erro_por_feature.head(20).sort_values().plot(kind="barh", ax=ax)
ax.set_title("Top 20 features com maior erro médio de reconstrução")
ax.set_xlabel("Erro absoluto médio")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "05_erro_reconstrucao_por_feature.png", dpi=180)
fig.savefig(IMAGENS_DIR / "05_erro_reconstrucao_por_feature.pdf")
plt.close(fig)

metadata = {
    "n_componentes": int(pca_model.n_components_),
    "variancia_retida": float(pca_model.explained_variance_ratio_.sum()),
    "erro_reconstrucao_mse": float(mse_reconstrucao),
}
(ARTEFACTOS_DIR / "pca_reconstrucao_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nMétricas de reconstrução:")
print(metricas)
print("\nImagens guardadas:")
print("- imagens/05_pca_2d_tipo_manobra.png")
print("- imagens/05_erro_reconstrucao_por_feature.png")
print("\nInterpretação:")
print("- Quanto menor o MSE de reconstrução, menor a perda de informação.")
print("- A visualização PC1/PC2 mostra apenas uma parte da variância total.")
print("- Se as classes se misturam no PCA 2D, isso não significa que o PCA falhou; significa que os dois primeiros componentes não separam claramente o rótulo.")
print("\nEtapa 5 concluída com sucesso.")
