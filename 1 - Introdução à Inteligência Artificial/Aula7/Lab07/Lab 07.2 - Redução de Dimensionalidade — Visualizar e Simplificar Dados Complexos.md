# Lab 07.2 - Redução de Dimensionalidade — Visualizar e Simplificar Dados Complexos

**Tema:** Redução de Dimensionalidade (PCA, Kernel PCA, LDA)

## 1. Objetivo

O objetivo deste laboratório é explorar técnicas de **redução de dimensionalidade** para dois fins principais:
1.  **Visualização:** Projetar dados de alta dimensão num espaço 2D ou 3D para que possamos "ver" a sua estrutura, identificar clusters ou anomalias visualmente.
2.  **Redução de Ruído/Features:** Simplificar o dataset, mantendo a maior parte da informação relevante, o que pode ajudar a acelerar o treino de outros modelos e a mitigar a "maldição da dimensionalidade".

## 2. Dados

Vais utilizar um dataset (`sensores_voo_alta_dim.csv`) que contém dezenas de leituras de sensores de um voo, resultando num espaço de características de alta dimensão.
- `sensor_01`, `sensor_02`, ..., `sensor_50`
- `tipo_manobra` (rótulo opcional, e.g., 'Cruzeiro', 'Subida', 'Descida') — Este rótulo **não** será usado para PCA/Kernel PCA, mas **será** usado para LDA.

## 3. Tarefas

1.  **Pré-processamento:**
    *   Como sempre, começa por **escalonar os dados** (`StandardScaler`). Isto é especialmente crítico para o PCA, que é sensível à variância das características.

2.  **Análise de Componentes Principais (PCA - Unsupervised):**
    *   Aplica o PCA ao teu dataset escalonado.
    *   Plota o gráfico da **variância explicada acumulada** em função do número de componentes.
    *   Quantos componentes são necessários para reter 95% da variância total dos dados?
    *   Cria um gráfico de dispersão 2D usando os dois primeiros componentes principais (PC1 e PC2). Consegues ver alguma estrutura ou agrupamento nos dados?
    *   (Opcional) Se tiveres os rótulos `tipo_manobra`, colore os pontos no gráfico 2D. O PCA conseguiu separar as classes, mesmo sem as ter visto?

3.  **Kernel PCA (para Não-Linearidade - Unsupervised):**
    *   Aplica o Kernel PCA, utilizando um kernel como o `rbf` (Radial Basis Function).
    *   Cria novamente um gráfico de dispersão 2D com os dois primeiros componentes.
    *   Compara esta visualização com a do PCA linear. O Kernel PCA revelou alguma estrutura que o PCA linear não conseguiu capturar?

4.  **Análise Discriminante Linear (LDA - Supervised):**
    *   **Nota:** LDA é uma técnica *supervisionada*. Ela usa os rótulos (`tipo_manobra`) para encontrar os eixos que **maximizam a separação** entre as classes.
    *   Aplica o LDA ao teu dataset, usando a variável `tipo_manobra` como alvo.
    *   Cria um gráfico de dispersão 2D usando os dois primeiros discriminantes lineares.
    *   Compara a separação das classes neste gráfico com a que obtiveste no gráfico do PCA.

## 4. Desafios e Lições

-   **Perda de Informação:** A redução de dimensionalidade é sempre um compromisso. Discute a perda de informação ao projetar os dados e como o gráfico de variância explicada ajuda a quantificar essa perda.
-   **Linear vs. Não-Linear:** Compara as projeções do PCA e do Kernel PCA. Em que cenários um seria mais útil que o outro?
-   **PCA (Unsupervised) vs. LDA (Supervised):** Reflete sobre a diferença fundamental entre os dois. O PCA procura os eixos de máxima variância nos dados, ignorando os rótulos. O LDA procura os eixos de máxima separabilidade entre classes conhecidas. São ferramentas para objetivos diferentes.

## Produto Final

Um relatório que apresente:  
-   A justificação para o escalonamento dos dados.  
-   O gráfico de variância explicada do PCA e a escolha do número de componentes.  
-   Três gráficos de dispersão 2D/3D lado a lado:  
    1.  A projeção do PCA.  
    2.  A projeção do Kernel PCA.  
    3.  A projeção do LDA.  
-   Uma análise comparativa das três visualizações, destacando os pontos fortes e fracos de cada técnica no contexto deste dataset.  