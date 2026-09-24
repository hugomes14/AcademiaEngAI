# 06_visualizacao_nao_linear_lda.py
# Lab 07.2 — Redução de Dimensionalidade
# Etapa 6: Kernel PCA, t-SNE, UMAP e LDA.
#
# Nota pedagógica:
# - Este ficheiro corre de forma direta, sem funções e sem main.
# - Kernel PCA é uma alternativa não linear ao PCA linear.
# - LDA é supervisionado: usa tipo_manobra para procurar eixos de separação entre classes.
# - t-SNE corre por defeito com uma amostra controlada.
# - UMAP é obrigatório neste laboratório e pertence às dependências por defeito.

from pathlib import Path
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
MODELOS_DIR = BASE_DIR / "modelos"
TABELAS_DIR = BASE_DIR / "tabelas"
IMAGENS_DIR = BASE_DIR / "imagens"
NUMBA_CACHE_DIR = BASE_DIR / ".numba_cache"

for directory in [ARTEFACTOS_DIR, MODELOS_DIR, TABELAS_DIR, IMAGENS_DIR, NUMBA_CACHE_DIR]:
    directory.mkdir(exist_ok=True)

os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

print("=" * 80)
print("LAB 07.2 — ETAPA 6: KERNEL PCA, t-SNE, UMAP E LDA")
print("=" * 80)

X_scaled_path = ARTEFACTOS_DIR / "X_scaled.csv"
y_path = ARTEFACTOS_DIR / "y.csv"
X_pca_path = ARTEFACTOS_DIR / "X_pca.csv"

for path in [X_scaled_path, y_path, X_pca_path]:
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro obrigatório em falta: {path}")

X_scaled = pd.read_csv(X_scaled_path)
y = pd.read_csv(y_path).iloc[:, 0].astype(str)
X_pca = pd.read_csv(X_pca_path)

print(f"X_scaled: {X_scaled.shape}")
print(f"Classes em y: {y.value_counts().to_dict()}")

# Para visualizações não lineares, uma amostra reduzida é suficiente em contexto didático.
# Isto evita tempos de execução longos em computadores mais modestos.
max_amostra = 500
if len(X_scaled) > max_amostra:
    sample_idx = X_scaled.sample(max_amostra, random_state=42).index
    X_vis = X_scaled.loc[sample_idx].reset_index(drop=True)
    y_vis = y.loc[sample_idx].reset_index(drop=True)
    print(f"Amostra usada para visualizações não lineares: {max_amostra} linhas")
else:
    X_vis = X_scaled.reset_index(drop=True)
    y_vis = y.reset_index(drop=True)
    print("Visualizações com o dataset completo.")

# -------------------------------------------------------------------------
# t-SNE
# -------------------------------------------------------------------------
executar_tsne = os.environ.get("EXECUTAR_TSNE", "1") != "0"
tsne_executado = False
tsne_mensagem = ""

if executar_tsne:
    try:
        print("\nA aplicar t-SNE...")
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, (len(X_vis) - 1) // 3))
        print(f"Perplexity usada no t-SNE: {perplexity}")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=750,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        X_tsne = tsne.fit_transform(X_vis)
        X_tsne_df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
        X_tsne_df["tipo_manobra"] = y_vis
        X_tsne_df.to_csv(ARTEFACTOS_DIR / "X_tsne.csv", index=False, encoding="utf-8")
        joblib.dump(tsne, MODELOS_DIR / "tsne_model.pkl")

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=X_tsne_df, x="TSNE1", y="TSNE2", hue="tipo_manobra", alpha=0.85, ax=ax)
        ax.set_title("Visualização t-SNE 2D")
        ax.legend(title="Tipo de manobra", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(IMAGENS_DIR / "06_tsne_2d_tipo_manobra.png", dpi=180)
        fig.savefig(IMAGENS_DIR / "06_tsne_2d_tipo_manobra.pdf")
        plt.close(fig)

        tsne_executado = True
        tsne_mensagem = "t-SNE executado com sucesso."
    except Exception as exc:
        tsne_mensagem = f"t-SNE não foi executado. Motivo: {exc}"
else:
    tsne_mensagem = "t-SNE foi desativado por EXECUTAR_TSNE=0."

if not tsne_executado:
    X_tsne_df = pd.DataFrame({"TSNE1": [], "TSNE2": [], "tipo_manobra": []})
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "t-SNE não executado\nVer metadata da etapa",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.set_title("Visualização t-SNE — etapa opcional")
    fig.tight_layout()
    fig.savefig(IMAGENS_DIR / "06_tsne_2d_tipo_manobra.png", dpi=180)
    fig.savefig(IMAGENS_DIR / "06_tsne_2d_tipo_manobra.pdf")
    plt.close(fig)
    pd.DataFrame(columns=["TSNE1", "TSNE2", "tipo_manobra"]).to_csv(
        ARTEFACTOS_DIR / "X_tsne.csv", index=False, encoding="utf-8"
    )

print("\nNota t-SNE:", tsne_mensagem)

# -------------------------------------------------------------------------
# Kernel PCA com RBF
# -------------------------------------------------------------------------
print("\nA aplicar Kernel PCA com kernel RBF...")
kernel_pca = KernelPCA(n_components=2, kernel="rbf", gamma=None, random_state=42)
X_kpca = kernel_pca.fit_transform(X_vis)
X_kpca_df = pd.DataFrame(X_kpca, columns=["KPC1", "KPC2"])
X_kpca_df["tipo_manobra"] = y_vis
X_kpca_df.to_csv(ARTEFACTOS_DIR / "X_kernel_pca.csv", index=False, encoding="utf-8")
joblib.dump(kernel_pca, MODELOS_DIR / "kernel_pca_rbf.pkl")

fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(data=X_kpca_df, x="KPC1", y="KPC2", hue="tipo_manobra", alpha=0.85, ax=ax)
ax.set_title("Visualização Kernel PCA RBF 2D")
ax.legend(title="Tipo de manobra", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "06_kernel_pca_rbf_2d_tipo_manobra.png", dpi=180)
fig.savefig(IMAGENS_DIR / "06_kernel_pca_rbf_2d_tipo_manobra.pdf")
plt.close(fig)

# -------------------------------------------------------------------------
# UMAP obrigatório
# -------------------------------------------------------------------------
umap_executado = False
umap_mensagem = ""

try:
    import umap  # type: ignore
except ImportError as exc:
    print("\nErro: a dependência obrigatória 'umap-learn' não está instalada.")
    print("Instala as dependências com: python -m pip install -r requirements.txt")
    raise SystemExit(1) from exc

print("\nA aplicar UMAP...")
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_vis)
X_umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
X_umap_df["tipo_manobra"] = y_vis
X_umap_df.to_csv(ARTEFACTOS_DIR / "X_umap.csv", index=False, encoding="utf-8")
joblib.dump(umap_model, MODELOS_DIR / "umap_model.pkl")

fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(data=X_umap_df, x="UMAP1", y="UMAP2", hue="tipo_manobra", alpha=0.85, ax=ax)
ax.set_title("Visualização UMAP 2D")
ax.legend(title="Tipo de manobra", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "06_umap_2d_tipo_manobra.png", dpi=180)
fig.savefig(IMAGENS_DIR / "06_umap_2d_tipo_manobra.pdf")
plt.close(fig)

umap_executado = True
umap_mensagem = "UMAP executado com sucesso."

# -------------------------------------------------------------------------
# LDA supervisionado
# -------------------------------------------------------------------------
print("\nA aplicar LDA supervisionado...")
le = LabelEncoder()
y_encoded = le.fit_transform(y_vis)

n_classes = len(le.classes_)
n_components_lda = min(2, n_classes - 1)

if n_components_lda < 1:
    raise ValueError("LDA precisa de pelo menos duas classes no rótulo.")

lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_lda = lda.fit_transform(X_vis, y_encoded)

if n_components_lda == 1:
    X_lda_df = pd.DataFrame({"LD1": X_lda[:, 0], "LD2": 0.0})
else:
    X_lda_df = pd.DataFrame(X_lda, columns=["LD1", "LD2"])

X_lda_df["tipo_manobra"] = y_vis
X_lda_df.to_csv(ARTEFACTOS_DIR / "X_lda.csv", index=False, encoding="utf-8")
joblib.dump(lda, MODELOS_DIR / "lda_model.pkl")
joblib.dump(le, MODELOS_DIR / "label_encoder_tipo_manobra.pkl")

fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(data=X_lda_df, x="LD1", y="LD2", hue="tipo_manobra", alpha=0.85, ax=ax)
ax.set_title("Visualização LDA 2D — usa o rótulo tipo_manobra")
ax.legend(title="Tipo de manobra", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(IMAGENS_DIR / "06_lda_2d_tipo_manobra.png", dpi=180)
fig.savefig(IMAGENS_DIR / "06_lda_2d_tipo_manobra.pdf")
plt.close(fig)

# -------------------------------------------------------------------------
# Figura comparativa
# -------------------------------------------------------------------------
pca_plot = X_pca[["PC1", "PC2"]].copy()
pca_plot["tipo_manobra"] = y.reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

sns.scatterplot(data=pca_plot, x="PC1", y="PC2", hue="tipo_manobra", alpha=0.75, ax=axes[0, 0], legend=False)
axes[0, 0].set_title("PCA")

if tsne_executado:
    sns.scatterplot(data=X_tsne_df, x="TSNE1", y="TSNE2", hue="tipo_manobra", alpha=0.75, ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("t-SNE")
else:
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, "t-SNE não executado\nver metadata", ha="center", va="center", fontsize=13)
    axes[0, 1].set_title("t-SNE")

sns.scatterplot(data=X_kpca_df, x="KPC1", y="KPC2", hue="tipo_manobra", alpha=0.75, ax=axes[1, 0], legend=False)
axes[1, 0].set_title("Kernel PCA RBF")

sns.scatterplot(data=X_lda_df, x="LD1", y="LD2", hue="tipo_manobra", alpha=0.75, ax=axes[1, 1])
axes[1, 1].set_title("LDA")
axes[1, 1].legend(title="Tipo de manobra", bbox_to_anchor=(1.02, 1), loc="upper left")

fig.suptitle("Comparação das projeções 2D", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(IMAGENS_DIR / "06_comparacao_projecoes_2d.png", dpi=180)
fig.savefig(IMAGENS_DIR / "06_comparacao_projecoes_2d.pdf")
plt.close(fig)

resumo = pd.DataFrame([
    {
        "metodo": "PCA",
        "tipo": "Linear, não supervisionado",
        "usa_rotulo_no_treino": "Não",
        "artefacto": "artefactos/X_pca.csv",
        "imagem": "imagens/05_pca_2d_tipo_manobra.png"
    },
    {
        "metodo": "t-SNE",
        "tipo": "Não linear, não supervisionado",
        "usa_rotulo_no_treino": "Não",
        "artefacto": "artefactos/X_tsne.csv" if tsne_executado else "Não gerado",
        "imagem": "imagens/06_tsne_2d_tipo_manobra.png"
    },
    {
        "metodo": "Kernel PCA RBF",
        "tipo": "Não linear, não supervisionado",
        "usa_rotulo_no_treino": "Não",
        "artefacto": "artefactos/X_kernel_pca.csv",
        "imagem": "imagens/06_kernel_pca_rbf_2d_tipo_manobra.png"
    },
    {
        "metodo": "UMAP",
        "tipo": "Não linear, não supervisionado",
        "usa_rotulo_no_treino": "Não",
        "artefacto": "artefactos/X_umap.csv" if umap_executado else "Não gerado",
        "imagem": "imagens/06_umap_2d_tipo_manobra.png" if umap_executado else "Não gerada"
    },
    {
        "metodo": "LDA",
        "tipo": "Linear, supervisionado",
        "usa_rotulo_no_treino": "Sim",
        "artefacto": "artefactos/X_lda.csv",
        "imagem": "imagens/06_lda_2d_tipo_manobra.png"
    },
])
resumo.to_csv(TABELAS_DIR / "06_resumo_metodos_visualizacao.csv", index=False, encoding="utf-8")
resumo.to_markdown(TABELAS_DIR / "06_resumo_metodos_visualizacao.md", index=False)

metadata = {
    "tsne_executado": tsne_executado,
    "tsne_mensagem": tsne_mensagem,
    "kernel_pca_kernel": "rbf",
    "umap_executado": umap_executado,
    "umap_mensagem": umap_mensagem,
    "lda_classes": le.classes_.tolist(),
    "lda_n_components": int(n_components_lda),
    "amostra_visualizacao": int(len(X_vis)),
}
(ARTEFACTOS_DIR / "visualizacoes_nao_lineares_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nResumo dos métodos:")
print(resumo)
print("\nImagens guardadas:")
print("- imagens/06_tsne_2d_tipo_manobra.png")
print("- imagens/06_kernel_pca_rbf_2d_tipo_manobra.png")
print("- imagens/06_lda_2d_tipo_manobra.png")
print("- imagens/06_comparacao_projecoes_2d.png")
if umap_executado:
    print("- imagens/06_umap_2d_tipo_manobra.png")
else:
    print("- UMAP não foi executado neste ambiente; ver metadata.")
print("\nEtapa 6 concluída com sucesso.")
