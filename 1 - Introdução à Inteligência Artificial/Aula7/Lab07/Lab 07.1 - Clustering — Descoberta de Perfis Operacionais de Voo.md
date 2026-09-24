# Lab 07.1 - Clustering — Descoberta de Perfis Operacionais de Voo

**Tema:** Aprendizagem Não Supervisionada - Clustering

## 1. Objetivo

Neste laboratório, o objetivo é usar algoritmos de **clustering** para identificar e analisar diferentes **perfis operacionais de voo** a partir de dados de telemetria. Em vez de prever um valor conhecido, vamos deixar que o algoritmo agrupe os voos com características semelhantes, descobrindo padrões que não são óbvios à primeira vista.

## 2. Dados

Vais trabalhar com um dataset (`voos_telemetria_completa.csv`) que contém métricas agregadas por voo:
- `duracao_voo_min`
- `distancia_percorrida_km`
- `altitude_maxima_m`
- `velocidade_media_kmh`
- `consumo_combustivel_litros`
- `variacao_vertical_total_m`

Nota que **não há uma variável alvo**. A nossa tarefa é criar os próprios grupos.

## 3. Tarefas

1.  **Pré-processamento e Análise:**
    *   Analisa a distribuição e a escala de cada variável.
    *   **Fundamental:** Normaliza/Escalona os dados (e.g., usando `StandardScaler` ou `MinMaxScaler`). Por que isto é crucial para algoritmos baseados em distância como o K-means?
    *   Visualiza as relações entre pares de variáveis (e.g., com um `pairplot`) para ter uma intuição inicial sobre possíveis agrupamentos.

2.  **Modelagem com K-means:**
    *   **Encontrar o 'k' ideal:** Usa o **Método do Cotovelo (Elbow Method)** para visualizar a inércia (WCSS) em função do número de clusters (k).
    *   Usa o **Coeficiente de Silhueta (Silhouette Score)** para avaliar a qualidade dos clusters para diferentes valores de 'k'.
    *   Escolhe o melhor 'k' com base nestas duas técnicas.
    *   Treina o modelo K-means final com o 'k' escolhido e atribui um rótulo de cluster a cada voo no dataset.

3.  **Modelagem com Outros Algoritmos (Comparação):**
    *   **Clustering Hierárquico Aglomerativo:**
        *   Cria um **dendrograma** para visualizar a estrutura hierárquica dos agrupamentos. Isto pode ajudar a confirmar a escolha de 'k'.
        *   Treina o modelo e compara os clusters resultantes com os do K-means.
    *   **(Opcional) Gaussian Mixture Models (GMM):**
        *   Treina um modelo GMM. Ao contrário do K-means, que assume clusters esféricos, o GMM pode capturar grupos com formas mais elípticas. Compara os resultados.

4.  **Análise e Interpretação dos Clusters:**
    *   Esta é a parte mais importante. Depois de formar os clusters, analisa as características de cada um.
    *   Calcula a média de cada variável para cada cluster (e.g., usando `groupby().mean()`).
    *   Cria "personas" ou perfis para cada cluster. Por exemplo:
        *   *Cluster 0: "Voos Curtos e de Baixa Altitude"*
        *   *Cluster 1: "Missões Longas e de Alta Velocidade"*
        *   *Cluster 2: "Voos com Muita Manobra Vertical"*

## 4. Desafios e Lições

-   **A Importância da Normalização:** Compara os resultados do K-means com e sem normalização para ver o impacto dramático que a escala das variáveis pode ter.
-   **A Subjetividade do 'k':** Discute como a escolha do número de clusters (k) nem sempre tem uma resposta única e pode depender do objetivo do negócio.
-   **Lidar com Outliers:** Como os outliers podem afetar o centróide de um cluster no K-means? Reflete sobre estratégias para lidar com eles (e.g., remoção ou uso de algoritmos mais robustos como o DBSCAN).

## Produto Final

Um relatório que apresente:  
-   O processo de pré-processamento, com ênfase na justificação da normalização.  
-   Os gráficos do Método do Cotovelo e da Pontuação de Silhueta.  
-   Uma comparação (pode ser visual, com gráficos de dispersão coloridos pelos clusters) dos resultados do K-means e do Clustering Hierárquico.  
-   Uma tabela e uma descrição detalhada que definam os **perfis de voo** para cada cluster descoberto.  