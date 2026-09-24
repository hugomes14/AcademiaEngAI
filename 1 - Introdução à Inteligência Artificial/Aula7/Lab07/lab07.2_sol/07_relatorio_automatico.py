# 07_relatorio_automatico.py
# Lab 07.2 — Redução de Dimensionalidade
# Etapa 7: geração automática do relatório final.
#
# Nota pedagógica:
# - Este ficheiro corre de forma direta, sem funções e sem main.
# - O relatório agrega tabelas, imagens e interpretações das etapas anteriores.

from pathlib import Path
import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RELATORIOS_DIR = BASE_DIR / "relatorios"
TABELAS_DIR = BASE_DIR / "tabelas"
IMAGENS_DIR = BASE_DIR / "imagens"
ARTEFACTOS_DIR = BASE_DIR / "artefactos"

RELATORIOS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("LAB 07.2 — ETAPA 7: RELATÓRIO AUTOMÁTICO")
print("=" * 80)

relatorio_path = RELATORIOS_DIR / "RELATORIO_FINAL.md"
relatorio_raiz_path = BASE_DIR / "RELATORIO_FINAL.md"

metadata_pre = {}
metadata_pca = {}
metadata_rec = {}
metadata_vis = {}

for path, destino in [
    (ARTEFACTOS_DIR / "preprocessamento_metadata.json", "metadata_pre"),
    (ARTEFACTOS_DIR / "pca_final_metadata.json", "metadata_pca"),
    (ARTEFACTOS_DIR / "pca_reconstrucao_metadata.json", "metadata_rec"),
    (ARTEFACTOS_DIR / "visualizacoes_nao_lineares_metadata.json", "metadata_vis"),
]:
    if path.exists():
        dados = json.loads(path.read_text(encoding="utf-8"))
        if destino == "metadata_pre":
            metadata_pre = dados
        elif destino == "metadata_pca":
            metadata_pca = dados
        elif destino == "metadata_rec":
            metadata_rec = dados
        elif destino == "metadata_vis":
            metadata_vis = dados

resumo_eda_md = ""
resumo_path = TABELAS_DIR / "01_resumo_estatistico_sensores.md"
if resumo_path.exists():
    resumo_eda_md = resumo_path.read_text(encoding="utf-8")

variancia_md = ""
variancia_path = TABELAS_DIR / "03_pca_variancia_componentes.md"
if variancia_path.exists():
    variancia_df = pd.read_csv(TABELAS_DIR / "03_pca_variancia_componentes.csv")
    # Para o relatório, mostra apenas os primeiros 15 componentes para ficar legível.
    variancia_reduzida = variancia_df.head(15).copy()
    variancia_reduzida["variancia_explicada_percentagem"] = variancia_reduzida["variancia_explicada_percentagem"].round(4)
    variancia_reduzida["variancia_acumulada_percentagem"] = variancia_reduzida["variancia_acumulada_percentagem"].round(4)
    variancia_reduzida["erro_reconstrucao_mse"] = variancia_reduzida["erro_reconstrucao_mse"].round(6)
    variancia_md = variancia_reduzida.to_markdown(index=False)

top_loadings_md = ""
top_loadings_path = TABELAS_DIR / "04_top_loadings_por_componente.md"
if top_loadings_path.exists():
    top_loadings_md = top_loadings_path.read_text(encoding="utf-8")

reconstrucao_md = ""
reconstrucao_path = TABELAS_DIR / "05_metricas_reconstrucao_pca.md"
if reconstrucao_path.exists():
    reconstrucao_md = reconstrucao_path.read_text(encoding="utf-8")

metodos_md = ""
metodos_path = TABELAS_DIR / "06_resumo_metodos_visualizacao.md"
if metodos_path.exists():
    metodos_md = metodos_path.read_text(encoding="utf-8")

n_linhas = metadata_pre.get("n_linhas", "n/d")
n_features = metadata_pre.get("n_features_processadas", "n/d")
target = metadata_pre.get("target_visualizacao", "tipo_manobra")
n_componentes = metadata_pca.get("n_components_95", "n/d")
variancia_retida = metadata_pca.get("variancia_total_retida", None)
erro_rec = metadata_rec.get("erro_reconstrucao_mse", None)
umap_msg = metadata_vis.get("umap_mensagem", "Informação sobre UMAP não disponível.")
umap_executado = metadata_vis.get("umap_executado", False)
bloco_umap_imagem = ""
bloco_umap_artefacto = ""

if umap_executado and (IMAGENS_DIR / "06_umap_2d_tipo_manobra.png").exists():
    bloco_umap_imagem = "\n![UMAP 2D](../imagens/06_umap_2d_tipo_manobra.png)\n"
    bloco_umap_artefacto = "\n- `artefactos/X_umap.csv`"

variancia_txt = f"{variancia_retida:.4f}" if isinstance(variancia_retida, float) else "n/d"
erro_txt = f"{erro_rec:.6f}" if isinstance(erro_rec, float) else "n/d"

relatorio = f"""# Relatório Final — Lab 07.2: Redução de Dimensionalidade

## 1. Introdução

Este relatório apresenta o fluxo completo do laboratório **Redução de Dimensionalidade — Visualizar e Simplificar Dados Complexos**.

O dataset contém leituras de sensores de voo de alta dimensão. O objetivo é reduzir a dimensionalidade para:

- visualizar a estrutura dos dados;
- identificar separações potenciais entre tipos de manobra;
- interpretar componentes principais;
- medir a perda de informação através do erro de reconstrução.

O rótulo `{target}` foi usado apenas para colorir visualizações e para LDA. PCA, t-SNE, Kernel PCA e UMAP não usam esse rótulo como variável de treino supervisionada.

## 2. Dataset

- Número de linhas: **{n_linhas}**
- Número de features processadas: **{n_features}**
- Rótulo opcional: **{target}**

## 3. Análise Exploratória

A análise exploratória avaliou distribuições, outliers, correlações e separação visual inicial.

### Imagens principais

![Histogramas dos sensores](../imagens/01_histogramas_sensores.png)

![Boxplots dos sensores](../imagens/01_boxplots_sensores.png)

![Heatmap de correlações](../imagens/01_heatmap_correlacoes.png)

![Pairplot dos sensores com maior variância](../imagens/01_pairplot_sensores_maior_variancia.png)

### Leitura pedagógica

O PCA é sensível à escala porque maximiza variância. Se os sensores tiverem escalas diferentes, um sensor com maior amplitude pode dominar artificialmente os componentes principais. Por isso, o escalonamento com `StandardScaler` é uma etapa obrigatória neste laboratório.

Outliers também merecem atenção. Como afetam média, variância e covariância, podem alterar a orientação dos componentes principais.

## 4. Pipeline de Pré-processamento

O pipeline aplicou:

1. separação entre `X` e o rótulo opcional `{target}`;
2. preservação do rótulo apenas para visualização e LDA;
3. imputação simples, se existirem valores em falta;
4. escalonamento com `StandardScaler`;
5. guarda de `X_scaled.csv`, `y.csv` e `scaler_standard.pkl`.

Artefactos principais:

- `artefactos/X_scaled.csv`
- `artefactos/y.csv`
- `modelos/scaler_standard.pkl`

## 5. PCA — Escolha do Número de Componentes

Foi treinado um PCA completo para medir a variância explicada por cada componente.

Componentes escolhidos para reter 95% da variância: **{n_componentes}**

Variância total retida pelo PCA final: **{variancia_txt}**

### Primeiros componentes

{variancia_md}

### Gráficos

![Scree Plot](../imagens/03_scree_plot_variancia.png)

![Variância acumulada](../imagens/03_variancia_acumulada.png)

![Erro de reconstrução por k](../imagens/03_erro_reconstrucao_por_k.png)

## 6. PCA — Loadings e Interpretação

Os loadings indicam o contributo de cada sensor para cada componente principal. Valores absolutos maiores indicam maior influência. O sinal positivo ou negativo indica a direção da relação com o componente.

![Heatmap de loadings](../imagens/04_heatmap_loadings_pca.png)

### Loadings mais relevantes

{top_loadings_md}

## 7. Visualização PCA 2D e Erro de Reconstrução

A projeção em PC1 e PC2 permite visualizar uma parte da estrutura dos dados. Como estes dois componentes não retêm toda a informação, a separação visual pode ser incompleta.

![PCA 2D](../imagens/05_pca_2d_tipo_manobra.png)

### Métrica de reconstrução

{reconstrucao_md}

Erro de reconstrução MSE: **{erro_txt}**

![Erro por feature](../imagens/05_erro_reconstrucao_por_feature.png)

## 8. Visualizações Não Lineares e LDA

Foram aplicadas técnicas complementares:

- **t-SNE**: útil para vizinhanças locais e visualização exploratória;
- **Kernel PCA RBF**: versão não linear do PCA;
- **UMAP**: método não linear obrigatório neste laboratório, executado com `umap-learn`;
- **LDA**: supervisionado, usa o rótulo para maximizar separação entre classes.

Nota sobre UMAP: {umap_msg}

### Resumo dos métodos

{metodos_md}

### Gráficos

![t-SNE 2D](../imagens/06_tsne_2d_tipo_manobra.png)

![Kernel PCA RBF 2D](../imagens/06_kernel_pca_rbf_2d_tipo_manobra.png)

{bloco_umap_imagem}

![LDA 2D](../imagens/06_lda_2d_tipo_manobra.png)

![Comparação das projeções](../imagens/06_comparacao_projecoes_2d.png)

## 9. Comparação dos Métodos

### PCA

PCA é linear, interpretável e permite medir a variância explicada. É adequado para compressão, redução de ruído e interpretação global.

### Kernel PCA

Kernel PCA pode revelar estruturas não lineares que o PCA linear não capta. A interpretação direta é mais difícil, porque os componentes deixam de ser combinações lineares simples das features originais.

### t-SNE

t-SNE ajuda a visualizar vizinhanças locais. É excelente para exploração visual, mas não preserva necessariamente distâncias globais. Não deve servir como prova final de separação entre grupos.

### UMAP

UMAP costuma produzir visualizações compactas e úteis, com melhor equilíbrio entre estrutura local e global. Requer a biblioteca `umap-learn`, incluída nas dependências deste laboratório.

### LDA

LDA é supervisionado. Ao contrário do PCA, usa os rótulos para encontrar eixos que maximizam a separação entre classes. Por isso, uma projeção LDA mais separada do que PCA não significa que o PCA falhou; significa que os métodos têm objetivos diferentes.

## 10. Conclusões e Recomendações

- O escalonamento é obrigatório antes do PCA, porque sensores com maior escala poderiam dominar a análise.
- A escolha do número de componentes deve equilibrar compressão e perda de informação.
- O gráfico de variância acumulada permite justificar a escolha de componentes para reter 95% da informação.
- Os loadings ajudam a explicar quais sensores mais influenciam os componentes principais.
- t-SNE, Kernel PCA e UMAP são úteis para visualização não linear, mas têm menor interpretabilidade do que PCA.
- LDA deve ser usado quando existe rótulo e o objetivo é maximizar separação entre classes.
- Como próximo passo, pode testar-se a influência da redução de dimensionalidade num modelo supervisionado, por exemplo classificar `tipo_manobra` com e sem PCA.

## 11. Ficheiros Gerados

### Modelos e transformadores

- `modelos/scaler_standard.pkl`
- `modelos/pca_full.pkl`
- `modelos/pca_model_95.pkl`
- `modelos/kernel_pca_rbf.pkl`
- `modelos/lda_model.pkl`
- `modelos/tsne_model.pkl`

### Dados transformados

- `artefactos/X_scaled.csv`
- `artefactos/X_pca.csv`
- `artefactos/X_tsne.csv`
- `artefactos/X_kernel_pca.csv`
{bloco_umap_artefacto}
- `artefactos/X_lda.csv`
- `artefactos/y.csv`

### Tabelas

- `tabelas/03_pca_variancia_componentes.csv`
- `tabelas/04_pca_loadings.csv`
- `tabelas/05_metricas_reconstrucao_pca.csv`
- `tabelas/06_resumo_metodos_visualizacao.csv`

## 12. Referências

- Scikit-learn: PCA, KernelPCA, TSNE e LinearDiscriminantAnalysis
- Conceitos: variância explicada, erro de reconstrução, loadings, projeções não lineares e separabilidade supervisionada
"""

relatorio_path.write_text(relatorio, encoding="utf-8")
relatorio_raiz_path.write_text(relatorio, encoding="utf-8")

print(f"Relatório guardado em: {relatorio_path}")
print(f"Cópia do relatório guardada em: {relatorio_raiz_path}")
print("Etapa 7 concluída com sucesso.")
