# Relatório Final — Lab 07.1: Clustering — Perfis Operacionais de Voo

## 1. Introdução

Este relatório apresenta uma análise não supervisionada de voos com base em métricas agregadas de telemetria. O objetivo não é prever uma variável-alvo, mas descobrir perfis operacionais com características semelhantes.

## 2. Dataset

O dataset usado foi `voos_telemetria_completa.csv`, com as seguintes variáveis:

- `duracao_voo_min`

- `distancia_percorrida_km`

- `altitude_maxima_m`

- `velocidade_media_kmh`

- `consumo_combustivel_litros`

- `variacao_vertical_total_m`


## 3. Análise Exploratória

A análise exploratória confirmou que as variáveis têm escalas muito diferentes. Este ponto é crítico, porque K-Means, DBSCAN e clustering hierárquico usam distâncias entre observações.


### Estatística descritiva

|                            |   count |     mean |      std |     min |     25% |      50% |      75% |       max |   skewness |
|:---------------------------|--------:|---------:|---------:|--------:|--------:|---------:|---------:|----------:|-----------:|
| duracao_voo_min            | 300.000 |   63.473 |   43.530 |  11.901 |  26.802 |   45.422 |  106.439 |   171.467 |      0.774 |
| distancia_percorrida_km    | 300.000 |   89.801 |   81.101 |  10.406 |  22.369 |   49.697 |  178.388 |   278.971 |      0.745 |
| altitude_maxima_m          | 300.000 | 2110.791 | 2181.590 |  52.762 | 172.224 | 1086.249 | 4741.235 |  6263.466 |      0.656 |
| velocidade_media_kmh       | 300.000 |  116.594 |   56.337 |  28.761 |  56.881 |  118.172 |  165.765 |   234.081 |      0.143 |
| consumo_combustivel_litros | 300.000 |   20.573 |   15.998 |   2.698 |   5.664 |   15.272 |   36.390 |    57.138 |      0.672 |
| variacao_vertical_total_m  | 300.000 | 3136.542 | 3620.862 | 176.418 | 326.129 | 1038.099 | 7196.293 | 11256.414 |      0.839 |



### Outliers pelo método IQR

| feature                    |      q1 |       q3 |      iqr |   limite_inferior |   limite_superior |   n_outliers_iqr |   percent_outliers_iqr |
|:---------------------------|--------:|---------:|---------:|------------------:|------------------:|-----------------:|-----------------------:|
| duracao_voo_min            |  26.802 |  106.439 |   79.637 |           -92.654 |           225.895 |                0 |                  0.000 |
| distancia_percorrida_km    |  22.369 |  178.388 |  156.019 |          -211.660 |           412.417 |                0 |                  0.000 |
| altitude_maxima_m          | 172.224 | 4741.235 | 4569.011 |         -6681.293 |         11594.751 |                0 |                  0.000 |
| velocidade_media_kmh       |  56.881 |  165.765 |  108.884 |          -106.445 |           329.091 |                0 |                  0.000 |
| consumo_combustivel_litros |   5.664 |   36.390 |   30.727 |           -40.427 |            82.481 |                0 |                  0.000 |
| variacao_vertical_total_m  | 326.129 | 7196.293 | 6870.164 |         -9979.117 |         17501.539 |                0 |                  0.000 |



### Figuras da EDA

- Histogramas e boxplots: `outputs/figures/eda/`

- Pairplot: `outputs/figures/eda/pairplot_variaveis.png`

- Correlações: `outputs/figures/eda/heatmap_correlacoes.png`


## 4. Pré-processamento

Foi aplicado `StandardScaler` ao dataset completo, porque não existe variável-alvo nem divisão treino/teste. O escalonamento impede que variáveis com unidades maiores, como altitude ou variação vertical, dominem as distâncias.


Artefactos principais:

- `outputs/tables/dados_limpos.csv`

- `outputs/tables/dados_processados_sem_escala.csv`

- `outputs/tables/dados_escalonados.csv`

- `outputs/models/standard_scaler.pkl`


## 5. Método Elbow e Escolha de k

O `k` recomendado automaticamente pela Silhouette foi: **3**.


### Métricas por k

|       k |   inercia_wcss |   silhouette |   davies_bouldin |   calinski_harabasz |
|--------:|---------------:|-------------:|-----------------:|--------------------:|
|  2.0000 |       481.2938 |       0.6767 |           0.5053 |            816.4961 |
|  3.0000 |       128.8076 |       0.7634 |           0.3330 |           1926.6886 |
|  4.0000 |       111.2396 |       0.5905 |           1.0575 |           1497.8877 |
|  5.0000 |       100.5259 |       0.5871 |           1.0534 |           1246.8046 |
|  6.0000 |        91.9184 |       0.4203 |           1.4388 |           1092.6558 |
|  7.0000 |        83.3638 |       0.4329 |           1.2852 |           1005.5813 |
|  8.0000 |        75.4927 |       0.4373 |           1.2545 |            952.8944 |
|  9.0000 |        70.4363 |       0.4383 |           1.2397 |            893.1878 |
| 10.0000 |        66.2723 |       0.4389 |           1.2116 |            842.9555 |



Figuras geradas:

- `outputs/figures/elbow/metodo_elbow_inercia.png`

- `outputs/figures/elbow/silhouette_por_k.png`

- `outputs/figures/elbow/davies_bouldin_por_k.png`

- `outputs/figures/elbow/calinski_harabasz_por_k.png`


## 6. Modelos de Clustering

Foram treinados três modelos:

- K-Means

- DBSCAN

- AgglomerativeClustering


As métricas internas usadas foram:

- **Silhouette Score**: maior é melhor.

- **Davies-Bouldin Index**: menor é melhor.

- **Calinski-Harabasz Index**: maior é melhor.


### Tabela comparativa

| modelo        | parametros                |   n_labels_total |   n_clusters_sem_ruido |   n_ruido |   silhouette |   davies_bouldin |   calinski_harabasz |
|:--------------|:--------------------------|-----------------:|-----------------------:|----------:|-------------:|-----------------:|--------------------:|
| kmeans        | k=3                       |                3 |                      3 |         0 |       0.7634 |           0.3330 |           1926.6886 |
| agglomerative | k=3                       |                3 |                      3 |         0 |       0.7634 |           0.3330 |           1926.6886 |
| dbscan        | eps=0.8727; min_samples=7 |                4 |                      3 |         1 |       0.5997 |           0.4634 |           1301.6092 |



Melhor modelo por Silhouette: **kmeans**.


## 7. Visualização PCA

A PCA foi usada apenas para visualização em 2D. As métricas foram calculadas no espaço completo escalonado.


Figuras principais:

- `outputs/figures/pca/pca_melhor_modelo.png`

- `outputs/figures/pca/pca_kmeans.png`

- `outputs/figures/pca/pca_agglomerative.png`

- `outputs/figures/pca/pca_dbscan.png`


## 8. DBSCAN e Outliers

O DBSCAN permite identificar observações como ruído (`label = -1`). Isto é útil quando alguns voos não encaixam bem nos perfis dominantes.


Figura principal:

- `outputs/figures/perfil_clusters/pca_dbscan_ruido_destacado.png`


## 9. Perfil dos Clusters

O profiling foi feito com os dados originais, não escalonados, para manter as unidades interpretáveis.


### Médias por cluster

|   cluster |   n_voos |   duracao_voo_min |   distancia_percorrida_km |   altitude_maxima_m |   velocidade_media_kmh |   consumo_combustivel_litros |   variacao_vertical_total_m |
|----------:|---------:|------------------:|--------------------------:|--------------------:|-----------------------:|-----------------------------:|----------------------------:|
|     0.000 |  100.000 |            24.481 |                    20.112 |             151.947 |                 51.068 |                        4.944 |                     294.235 |
|     1.000 |  100.000 |           120.482 |                   199.815 |            5114.325 |                180.696 |                       41.281 |                    1021.418 |
|     2.000 |  100.000 |            45.457 |                    49.476 |            1066.102 |                118.017 |                       15.495 |                    8093.973 |



### Personas sugeridas

|   cluster | nome_sugerido                     |   n_voos | features_acima_da_media                                                                                       | features_abaixo_da_media                                                                                                                 | interpretacao                                                                                                                                                                                                                                                                                     |
|----------:|:----------------------------------|---------:|:--------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|         0 | Voos curtos e compactos           |      100 | nenhuma                                                                                                       | duracao_voo_min, distancia_percorrida_km, altitude_maxima_m, velocidade_media_kmh, consumo_combustivel_litros, variacao_vertical_total_m | O cluster 0 foi caracterizado como 'Voos curtos e compactos'. As variáveis acima da média global são: nenhuma. As variáveis abaixo da média global são: duracao_voo_min, distancia_percorrida_km, altitude_maxima_m, velocidade_media_kmh, consumo_combustivel_litros, variacao_vertical_total_m. |
|         1 | Missões longas e extensas         |      100 | duracao_voo_min, distancia_percorrida_km, altitude_maxima_m, velocidade_media_kmh, consumo_combustivel_litros | variacao_vertical_total_m                                                                                                                | O cluster 1 foi caracterizado como 'Missões longas e extensas'. As variáveis acima da média global são: duracao_voo_min, distancia_percorrida_km, altitude_maxima_m, velocidade_media_kmh, consumo_combustivel_litros. As variáveis abaixo da média global são: variacao_vertical_total_m.        |
|         2 | Voos com elevada manobra vertical |      100 | variacao_vertical_total_m                                                                                     | duracao_voo_min, distancia_percorrida_km, altitude_maxima_m, consumo_combustivel_litros                                                  | O cluster 2 foi caracterizado como 'Voos com elevada manobra vertical'. As variáveis acima da média global são: variacao_vertical_total_m. As variáveis abaixo da média global são: duracao_voo_min, distancia_percorrida_km, altitude_maxima_m, consumo_combustivel_litros.                      |



Figuras principais:

- `outputs/figures/perfil_clusters/perfil_clusters_medias_normalizadas.png`

- `outputs/figures/perfil_clusters/heatmap_perfil_clusters.png`


## 10. Conclusões

O laboratório demonstra que clustering não termina no treino do modelo. A etapa decisiva é interpretar os grupos e confirmar se têm utilidade operacional.


Pontos principais:

- O escalonamento é obrigatório em algoritmos baseados em distância.

- A escolha de `k` deve combinar métricas, gráficos e interpretação de negócio.

- Outliers podem alterar centróides no K-Means e devem ser analisados.

- DBSCAN é útil para detetar observações atípicas, mas é sensível aos parâmetros `eps` e `min_samples`.

- A PCA ajuda a comunicar resultados, mas não substitui as métricas calculadas no espaço original de features escalonadas.


## 11. Recomendações

- Testar outros valores de `k` próximos do recomendado.

- Comparar `StandardScaler` com `RobustScaler` se houver muitos outliers.

- Validar os perfis com especialistas de operação de voo.

- Experimentar GMM caso se suspeite de clusters elípticos.

- Repetir a análise com novas variáveis de telemetria, se existirem.


## 12. Referências

- scikit-learn: KMeans, DBSCAN, AgglomerativeClustering, PCA.

- Métricas internas de clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz.
