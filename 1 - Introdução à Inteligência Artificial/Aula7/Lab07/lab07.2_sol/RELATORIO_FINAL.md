# Relatório Final — Lab 07.2: Redução de Dimensionalidade

## 1. Introdução

Este relatório apresenta o fluxo completo do laboratório **Redução de Dimensionalidade — Visualizar e Simplificar Dados Complexos**.

O dataset contém leituras de sensores de voo de alta dimensão. O objetivo é reduzir a dimensionalidade para:

- visualizar a estrutura dos dados;
- identificar separações potenciais entre tipos de manobra;
- interpretar componentes principais;
- medir a perda de informação através do erro de reconstrução.

O rótulo `tipo_manobra` foi usado apenas para colorir visualizações e para LDA. PCA, t-SNE, Kernel PCA e UMAP não usam esse rótulo como variável de treino supervisionada.

## 2. Dataset

- Número de linhas: **600**
- Número de features processadas: **50**
- Rótulo opcional: **tipo_manobra**

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

1. separação entre `X` e o rótulo opcional `tipo_manobra`;
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

Componentes escolhidos para reter 95% da variância: **32**

Variância total retida pelo PCA final: **0.9530**

### Primeiros componentes

| componente   |   numero_componente |   variancia_explicada |   variancia_explicada_percentagem |   variancia_acumulada |   variancia_acumulada_percentagem |   erro_reconstrucao_mse |
|:-------------|--------------------:|----------------------:|----------------------------------:|----------------------:|----------------------------------:|------------------------:|
| PC1          |                   1 |             0.351522  |                           35.1522 |              0.351522 |                           35.1522 |                0.648478 |
| PC2          |                   2 |             0.0627092 |                            6.2709 |              0.414231 |                           41.4231 |                0.585769 |
| PC3          |                   3 |             0.0283422 |                            2.8342 |              0.442573 |                           44.2573 |                0.557427 |
| PC4          |                   4 |             0.0268143 |                            2.6814 |              0.469387 |                           46.9387 |                0.530613 |
| PC5          |                   5 |             0.0260901 |                            2.609  |              0.495477 |                           49.5477 |                0.504523 |
| PC6          |                   6 |             0.0253937 |                            2.5394 |              0.520871 |                           52.0871 |                0.479129 |
| PC7          |                   7 |             0.0248918 |                            2.4892 |              0.545763 |                           54.5763 |                0.454237 |
| PC8          |                   8 |             0.0235745 |                            2.3574 |              0.569337 |                           56.9337 |                0.430663 |
| PC9          |                   9 |             0.0228498 |                            2.285  |              0.592187 |                           59.2187 |                0.407813 |
| PC10         |                  10 |             0.0223481 |                            2.2348 |              0.614535 |                           61.4535 |                0.385465 |
| PC11         |                  11 |             0.0217061 |                            2.1706 |              0.636241 |                           63.6241 |                0.363759 |
| PC12         |                  12 |             0.0211434 |                            2.1143 |              0.657385 |                           65.7385 |                0.342615 |
| PC13         |                  13 |             0.0204083 |                            2.0408 |              0.677793 |                           67.7793 |                0.322207 |
| PC14         |                  14 |             0.0201781 |                            2.0178 |              0.697971 |                           69.7971 |                0.302029 |
| PC15         |                  15 |             0.0199941 |                            1.9994 |              0.717965 |                           71.7965 |                0.282035 |

### Gráficos

![Scree Plot](../imagens/03_scree_plot_variancia.png)

![Variância acumulada](../imagens/03_variancia_acumulada.png)

![Erro de reconstrução por k](../imagens/03_erro_reconstrucao_por_k.png)

## 6. PCA — Loadings e Interpretação

Os loadings indicam o contributo de cada sensor para cada componente principal. Valores absolutos maiores indicam maior influência. O sinal positivo ou negativo indica a direção da relação com o componente.

![Heatmap de loadings](../imagens/04_heatmap_loadings_pca.png)

### Loadings mais relevantes

| componente   | feature   |   loading |   abs_loading |
|:-------------|:----------|----------:|--------------:|
| PC1          | sensor_07 |  0.228772 |      0.228772 |
| PC1          | sensor_13 |  0.227767 |      0.227767 |
| PC1          | sensor_08 |  0.226804 |      0.226804 |
| PC1          | sensor_10 |  0.226296 |      0.226296 |
| PC1          | sensor_06 |  0.225793 |      0.225793 |
| PC1          | sensor_12 |  0.224574 |      0.224574 |
| PC1          | sensor_09 |  0.224285 |      0.224285 |
| PC1          | sensor_14 |  0.223868 |      0.223868 |
| PC1          | sensor_15 |  0.218908 |      0.218908 |
| PC1          | sensor_11 |  0.215641 |      0.215641 |
| PC2          | sensor_24 |  0.299212 |      0.299212 |
| PC2          | sensor_21 |  0.283007 |      0.283007 |
| PC2          | sensor_25 |  0.275108 |      0.275108 |
| PC2          | sensor_23 |  0.274857 |      0.274857 |
| PC2          | sensor_22 |  0.270617 |      0.270617 |
| PC2          | sensor_40 | -0.240877 |      0.240877 |
| PC2          | sensor_34 | -0.236291 |      0.236291 |
| PC2          | sensor_32 | -0.226272 |      0.226272 |
| PC2          | sensor_33 | -0.222165 |      0.222165 |
| PC2          | sensor_38 | -0.217816 |      0.217816 |
| PC3          | sensor_20 |  0.425364 |      0.425364 |
| PC3          | sensor_43 |  0.367253 |      0.367253 |
| PC3          | sensor_48 |  0.361688 |      0.361688 |
| PC3          | sensor_27 | -0.337499 |      0.337499 |
| PC3          | sensor_50 |  0.263193 |      0.263193 |
| PC3          | sensor_41 | -0.21879  |      0.21879  |
| PC3          | sensor_17 | -0.216892 |      0.216892 |
| PC3          | sensor_29 |  0.20638  |      0.20638  |
| PC3          | sensor_26 | -0.196553 |      0.196553 |
| PC3          | sensor_02 |  0.188483 |      0.188483 |
| PC4          | sensor_02 |  0.377646 |      0.377646 |
| PC4          | sensor_46 |  0.371365 |      0.371365 |
| PC4          | sensor_50 |  0.307472 |      0.307472 |
| PC4          | sensor_47 | -0.304811 |      0.304811 |
| PC4          | sensor_05 | -0.298701 |      0.298701 |
| PC4          | sensor_43 |  0.262832 |      0.262832 |
| PC4          | sensor_29 | -0.251289 |      0.251289 |
| PC4          | sensor_26 |  0.241037 |      0.241037 |
| PC4          | sensor_44 | -0.238046 |      0.238046 |
| PC4          | sensor_28 |  0.18091  |      0.18091  |
| PC5          | sensor_19 |  0.405583 |      0.405583 |
| PC5          | sensor_18 |  0.384872 |      0.384872 |
| PC5          | sensor_01 | -0.282281 |      0.282281 |
| PC5          | sensor_26 | -0.252372 |      0.252372 |
| PC5          | sensor_41 |  0.250046 |      0.250046 |
| PC5          | sensor_02 | -0.244881 |      0.244881 |
| PC5          | sensor_29 |  0.234784 |      0.234784 |
| PC5          | sensor_16 |  0.225544 |      0.225544 |
| PC5          | sensor_04 | -0.220361 |      0.220361 |
| PC5          | sensor_47 | -0.214494 |      0.214494 |
| PC6          | sensor_30 |  0.464069 |      0.464069 |
| PC6          | sensor_28 |  0.355904 |      0.355904 |
| PC6          | sensor_16 |  0.346678 |      0.346678 |
| PC6          | sensor_26 |  0.232343 |      0.232343 |
| PC6          | sensor_42 | -0.216764 |      0.216764 |
| PC6          | sensor_18 |  0.204943 |      0.204943 |
| PC6          | sensor_04 |  0.197899 |      0.197899 |
| PC6          | sensor_50 |  0.191183 |      0.191183 |
| PC6          | sensor_46 | -0.180452 |      0.180452 |
| PC6          | sensor_05 |  0.1771   |      0.1771   |
| PC7          | sensor_45 |  0.430627 |      0.430627 |
| PC7          | sensor_29 | -0.350211 |      0.350211 |
| PC7          | sensor_28 | -0.331737 |      0.331737 |
| PC7          | sensor_05 |  0.304387 |      0.304387 |
| PC7          | sensor_19 |  0.303939 |      0.303939 |
| PC7          | sensor_41 |  0.273843 |      0.273843 |
| PC7          | sensor_30 | -0.246137 |      0.246137 |
| PC7          | sensor_18 | -0.240113 |      0.240113 |
| PC7          | sensor_43 |  0.190966 |      0.190966 |
| PC7          | sensor_03 |  0.165931 |      0.165931 |
| PC8          | sensor_49 |  0.405352 |      0.405352 |
| PC8          | sensor_03 | -0.280133 |      0.280133 |
| PC8          | sensor_47 | -0.253588 |      0.253588 |
| PC8          | sensor_04 | -0.246392 |      0.246392 |
| PC8          | sensor_28 | -0.240277 |      0.240277 |
| PC8          | sensor_18 | -0.240049 |      0.240049 |
| PC8          | sensor_44 | -0.239702 |      0.239702 |
| PC8          | sensor_26 |  0.235421 |      0.235421 |
| PC8          | sensor_29 |  0.233901 |      0.233901 |
| PC8          | sensor_19 | -0.228155 |      0.228155 |
| PC9          | sensor_01 |  0.397049 |      0.397049 |
| PC9          | sensor_42 |  0.380747 |      0.380747 |
| PC9          | sensor_44 | -0.358993 |      0.358993 |
| PC9          | sensor_16 |  0.291919 |      0.291919 |
| PC9          | sensor_45 |  0.291422 |      0.291422 |
| PC9          | sensor_26 | -0.273686 |      0.273686 |
| PC9          | sensor_04 | -0.253338 |      0.253338 |
| PC9          | sensor_18 | -0.201427 |      0.201427 |
| PC9          | sensor_29 |  0.182679 |      0.182679 |
| PC9          | sensor_41 | -0.169303 |      0.169303 |
| PC10         | sensor_01 |  0.45295  |      0.45295  |
| PC10         | sensor_17 | -0.378411 |      0.378411 |
| PC10         | sensor_47 | -0.302652 |      0.302652 |
| PC10         | sensor_30 |  0.263812 |      0.263812 |
| PC10         | sensor_26 | -0.251178 |      0.251178 |
| PC10         | sensor_27 | -0.233765 |      0.233765 |
| PC10         | sensor_50 | -0.231715 |      0.231715 |
| PC10         | sensor_43 | -0.215835 |      0.215835 |
| PC10         | sensor_49 |  0.212095 |      0.212095 |
| PC10         | sensor_42 | -0.201574 |      0.201574 |

## 7. Visualização PCA 2D e Erro de Reconstrução

A projeção em PC1 e PC2 permite visualizar uma parte da estrutura dos dados. Como estes dois componentes não retêm toda a informação, a separação visual pode ser incompleta.

![PCA 2D](../imagens/05_pca_2d_tipo_manobra.png)

### Métrica de reconstrução

| modelo           |   n_componentes |   variancia_retida |   erro_reconstrucao_mse |
|:-----------------|----------------:|-------------------:|------------------------:|
| PCA_95_variancia |              32 |           0.953039 |               0.0469608 |

Erro de reconstrução MSE: **0.046961**

![Erro por feature](../imagens/05_erro_reconstrucao_por_feature.png)

## 8. Visualizações Não Lineares e LDA

Foram aplicadas técnicas complementares:

- **t-SNE**: útil para vizinhanças locais e visualização exploratória;
- **Kernel PCA RBF**: versão não linear do PCA;
- **UMAP**: método não linear obrigatório neste laboratório, executado com `umap-learn`;
- **LDA**: supervisionado, usa o rótulo para maximizar separação entre classes.

Nota sobre UMAP: UMAP executado com sucesso.

### Resumo dos métodos

| metodo         | tipo                           | usa_rotulo_no_treino   | artefacto                   | imagem                                        |
|:---------------|:-------------------------------|:-----------------------|:----------------------------|:----------------------------------------------|
| PCA            | Linear, não supervisionado     | Não                    | artefactos/X_pca.csv        | imagens/05_pca_2d_tipo_manobra.png            |
| t-SNE          | Não linear, não supervisionado | Não                    | artefactos/X_tsne.csv       | imagens/06_tsne_2d_tipo_manobra.png           |
| Kernel PCA RBF | Não linear, não supervisionado | Não                    | artefactos/X_kernel_pca.csv | imagens/06_kernel_pca_rbf_2d_tipo_manobra.png |
| UMAP           | Não linear, não supervisionado | Não                    | artefactos/X_umap.csv       | imagens/06_umap_2d_tipo_manobra.png           |
| LDA            | Linear, supervisionado         | Sim                    | artefactos/X_lda.csv        | imagens/06_lda_2d_tipo_manobra.png            |

### Gráficos

![t-SNE 2D](../imagens/06_tsne_2d_tipo_manobra.png)

![Kernel PCA RBF 2D](../imagens/06_kernel_pca_rbf_2d_tipo_manobra.png)


![UMAP 2D](../imagens/06_umap_2d_tipo_manobra.png)


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

- `artefactos/X_umap.csv`
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
