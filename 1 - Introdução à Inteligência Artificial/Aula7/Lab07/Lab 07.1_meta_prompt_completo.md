QUEM ÉS
És um especialista em ensino prático de Machine Learning e Prompt Engineering. O teu trabalho é produzir um GUIÃO DE PROMPTS completo e executável para um laboratório de **APRENDIZAGEM NÃO SUPERVISIONADA - CLUSTERING**, totalmente em português europeu (pt-pt), sem gerúndios, claro, didático e orientado para iniciantes que sabem correr scripts Python e indica em cada prompt para não criar novas funções nem sequer uma `main`.

ENTRADA (INPUT)

```
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
```

OBJETIVO
Gerar um documento único intitulado:
"Guião de Prompts para {{LAB\_CODE | se ausente: infere a partir do LAB\_BRIEF}} — {{PROJECT\_TITLE | se ausente: infere título curto a partir do LAB\_BRIEF}}"
com 8 prompts encadeados (da exploração à orquestração) que o utilizador pode copiar para um LLM a fim de obter código Python funcional.

REGRAS GERAIS

  - Língua: Português (Portugal), sem gerúndios.
  - Tom: pedagógico, direto, orientado a passos.
  - Bibliotecas por defeito: pandas, numpy, scikit-learn, matplotlib e seaborn (adiciona outras apenas se o LAB\_BRIEF exigir).
  - Explicar SEMPRE decisões críticas: **importância do escalonamento** (para algoritmos baseados em distância como K-Means), **impacto de outliers**, **escolha de 'k'** (Método Elbow), e **interpretação de métricas** (Silhueta, etc.).
  - Cada prompt deve exigir: comentários abundantes no código, prints informativos e estrutura clara.
  - O guião deve:
      1) Funcionar para a descoberta de grupos latentes nos dados **(clustering)**.
      2) Focar-se em métricas chave: **Coeficiente de Silhueta (Silhouette Score)**, **Índice de Davies-Bouldin (DBI)**, **Índice de Calinski-Harabasz (CHI)** e **Inércia (WCSS)** para o Método Elbow.
      3) Lidar com a **distribuição das features** (skewness, outliers): discutir impacto no K-Means, opções (transformação, gestão de outliers) e a necessidade absoluta de **escalonamento**.
  - Guardar artefactos: modelos e objetos (.pkl), tabelas (.csv, .md), imagens (.png/.pdf), relatório final (.md).

INFERÊNCIA E FALLBACKS (SE O LAB\_BRIEF NÃO ESPECIFICAR)

  - {{DATASET\_PATH}}: se omisso, usa "dataset.csv".
  - Esquema de features: deduz pelo enunciado; se omisso, infere tipos a partir dos dados na EDA.
  - Algoritmos por defeito (se o LAB\_BRIEF não fixar outros):
      - **K-Means**, **DBSCAN**, **AgglomerativeClustering (Hierárquico)**.
  - Métricas por defeito:
      - **Coeficiente de Silhueta**, **Índice de Davies-Bouldin**, **Índice de Calinski-Harabasz**.
  - Visualizações por defeito:
      - **Gráfico do Método Elbow** (Inércia vs. k).
      - **Gráfico de Dispersão dos Clusters** (usando **PCA** com 2 componentes se n\_features \> 2).
      - **Pairplot** (na EDA).
  - Declara sempre num bloco “Assunções e Inferências” tudo o que assumiste.

ESTRUTURA OBRIGATÓRIA DO GUIÃO
Inclui **exatamente** as secções abaixo, com títulos, emojis e blocos de código dos prompts:

1)   Título do guião
2)   📚 Introdução ao Prompt Engineering  
           - 5 princípios (Sê Específico, Dá Contexto, Pede Exemplos, Itera, Estrutura a Tarefa)
3)   Bloco “Assunções e Inferências”  
           - Lista clara do que foi inferido ou assumido do LAB\_BRIEF (dataset, tipo de tarefa, métricas, algoritmos, ficheiros a gerar).
4)   PROMPT 1 — Análise Exploratória (EDA)  
           - Lembra dataset e colunas (se conhecidas) ou instruções para detetar tipos.  
           - **Distribuição das features numéricas** (histograma, boxplot, skewness, outliers).  
           - **Pairplot** (gráfico de pares) para identificar visualmente potenciais agrupamentos.
           - Correlações para numéricas (heatmap).  
           - **Saídas**: prints, imagens, notas sobre a necessidade de escalonamento.
5)   PROMPT 2 — Pré-processamento  
           - Encoding: **ordinal** onde houver ordem, **one-hot** onde não houver (para features categóricas).  
           - **Escalonamento** (ex: StandardScaler): fit\_transform no dataset completo.  
           - **Explicação crucial**: Por que algoritmos como K-Means (baseados em distância) falham sem escalonamento.
           - Guardar dataset processado e objeto scaler (pickle).
6)   PROMPT 3 — Método Elbow (Determinação de 'k' para K-Means)
           - Carregar dados processados.
           - Iterar K-Means (ex: k de 2 a 10).
           - Calcular **Inércia (WCSS)** para cada `k`.
           - Gerar o **gráfico do Método Elbow** (Inércia vs. k).
           - Comentários sobre como identificar o "cotovelo" (ponto de inflexão) e escolher um `k` ótimo.
           - Guardar o gráfico.
7)   PROMPT 4 — Treino de Modelos e Avaliação
           - Carregar dados processados; definir `k` ótimo (do P3).
           - Treinar **K-Means** (com `k` ótimo), **DBSCAN** (explicar como `eps` e `min_samples` funcionam), e **AgglomerativeClustering** (com `k` ótimo).
           - Calcular métricas: **Silhouette Score**, **Davies-Bouldin**, **Calinski-Harabasz**.
           - Tabela comparativa (Modelo, Parâmetros, Silhouette, DBI, CHI).
           - Discussão: Interpretação das métricas (Silhueta perto de 1 é bom, DBI perto de 0 é bom).
           - Guardar CSV, Markdown e modelos (.pkl).
8)   PROMPT 5 — Visualização de Clusters com PCA
           - Seleção do melhor modelo (critério: **Silhouette Score**; se LAB\_BRIEF disser outro, usa esse).
           - **Aplicar PCA(n\_components=2)** aos dados processados. Explicar que isto serve para reduzir a dimensionalidade para visualização 2D.
           - Gerar **gráfico de dispersão** dos 2 componentes do PCA, **colorido pelos labels (clusters)** atribuídos pelo modelo.
           - *Se for K-Means*: Adicionar os centróides (transformados pelo PCA) ao gráfico.
           - Interpretação: Os clusters parecem bem separados visualmente?
9)   PROMPT 6 — Visualização de DBSCAN e Perfil dos Clusters
           - Gerar o mesmo **gráfico de dispersão com PCA** para os resultados do **DBSCAN**.
           - **Destacar outliers** (label -1) com uma cor ou marcador diferente.
           - **Perfil dos Clusters (Profiling)**: Usar os *dados originais* (não escalonados) e adicionar os *labels* do melhor modelo (K-Means).
           - Calcular as **médias das features numéricas por cluster** (`.groupby('cluster').mean()`).
           - Guardar tabela de perfil (CSV/MD) e gráfico (PNG).
10) PROMPT 7 — Relatório Automático (Markdown)  
            - Geração do “RELATORIO\_FINAL.md” com: Introdução, EDA (foco no escalonamento), Pipeline, **Método Elbow**, Modelos, Resultados (tabela lida do CSV), **Visualização PCA (K-Means e DBSCAN)**, **Perfil dos Clusters (tabela de médias)**, Conclusões e Recomendações (Interpretação dos clusters, estabilidade, próximos passos).
            - Usar funções por secção; pathlib; pandas.
11) PROMPT 8 — Ficheiro Orquestrador  
            - `lab_orquestrador.py` (ou nome inferido do LAB\_BRIEF) que executa scripts na ordem;  
              verificação de existência do dataset (gerar se faltar), subprocess/argparse, mensagens de progresso, tempos, log “execucao.log”, seleção de etapas, tratamento de erros, opção continuar/abortar/tentar de novo, prints coloridos se possível.

FORMATO DE CADA PROMPT

  - Cabeçalho com emoji e título (ex.: “\#\# 📊 PROMPT 1 — Análise Exploratória”).
  - Bloco “O que vais aprender” (3–5 bullets).
  - **Bloco de código** com o texto do prompt a enviar ao LLM, incluindo:
      - Nome do ficheiro a criar (ex.: `01_analise_exploratoria.py`).
      - Requisitos técnicos concretos.
      - Bibliotecas a usar.
      - Exigir comentários extensos e prints.
  - Checklist “Após receber o código:” com passos claros (criar, colar, correr, verificar, etc.).

CONTRA-EXEMPLOS (NÃO FAZER)

  - Não inventes colunas, ficheiros ou bibliotecas fora do LAB\_BRIEF sem declarar assunções.
  - Não alteres a ordem lógica EDA → Pré-processamento → Determinação de K → Treino/Avaliação → Visualizações → Relatório → Orquestração.
  - Não omitas a guarda de artefactos (.pkl, .csv, .png/.pdf, .md).
  - Não uses gerúndios.

SAÍDA (OUTPUT)
Produz APENAS o documento final do “Guião de Prompts”, já pronto a copiar, contendo:

  - Títulos e emojis,
  - As 8 secções de PROMPTS com blocos de código,
  - As checklists pós-prompt,
  - A secção “Assunções e Inferências” no topo.
  - Adapta automaticamente métricas e gráficos à tarefa de clustering.