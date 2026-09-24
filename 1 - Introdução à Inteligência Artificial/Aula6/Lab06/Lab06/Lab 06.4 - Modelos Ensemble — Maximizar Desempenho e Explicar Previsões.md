# Lab 06.4 - Modelos Ensemble — Maximizar Desempenho e Explicar Previsões

**Tema:** Métodos Ensemble (Bagging e Boosting)

## 1. Objetivo

O objetivo deste laboratório é duplo:
1.  Utilizar modelos **ensemble** (como Random Forest e Gradient Boosting) para melhorar o desempenho de previsão numa tarefa de classificação (ou regressão).
2.  Ir além da simples previsão e usar estes modelos para **interpretar os dados**, identificando as variáveis mais importantes e como elas influenciam o resultado.

## 2. Dados

Vais usar o mesmo dataset de classificação do Lab 6.2 (`voos_pre_voo.csv`), focado em prever `incidente_reportado` (0 ou 1). Isto permite uma comparação direta do poder preditivo dos ensembles com os modelos mais simples.

**Recordatório dos Dados:**
- `idade_aeronave_anos`, `horas_voo_desde_ultima_manutencao`, `previsao_turbulencia`, `tipo_missao`, `experiencia_piloto_anos`.
- **Alvo:** `incidente_reportado`.

## 3. Tarefas

1.  **Modelo Base: Árvore de Decisão**
    *   Treina uma única **Árvore de Decisão** (sem limite de profundidade) para servir como *baseline*.
    *   Visualiza a árvore. Observa como ela pode facilmente sofrer *overfitting*.
    *   Avalia o seu desempenho no conjunto de teste (usando F1-Score e ROC-AUC).

2.  **Modelagem com Ensembles:**
    *   **Random Forest (Bagging):** Treina um modelo `RandomForestClassifier`.
        *   Analisa a métrica **Out-of-Bag (OOB) score**. Como ela se compara à pontuação no conjunto de teste?
    *   **XGBoost (Boosting):** Treina um modelo `XGBClassifier`.
        *   Utiliza a validação cruzada para encontrar um bom número de estimadores (`n_estimators`).
    *   **(Opcional) CatBoost:** Treina um `CatBoostClassifier`, que é conhecido por lidar bem com variáveis categóricas de forma nativa. Compara a sua conveniência e desempenho.

3.  **Avaliação e Comparação:**
    *   Compara as métricas de desempenho (F1, ROC-AUC) da Árvore de Decisão, Random Forest e XGBoost.
    *   Qual modelo obteve o melhor resultado? Houve um ganho significativo em relação aos modelos do Lab 6.2?

4.  **Explicabilidade (XAI - Explainable AI):**
    *   **Importância de Variáveis (Feature Importance):**
        *   Extrai e plota a importância das variáveis a partir do Random Forest e do XGBoost.
        *   As duas listas são consistentes? Quais são as 3 variáveis mais influentes?
    *   **Análise de Efeitos Parciais (Partial Dependence Plots - PDP):**
        *   Gera um PDP para as 1 ou 2 variáveis mais importantes.
        *   O que o PDP nos diz sobre a *relação* entre essa variável e a probabilidade de um incidente? (e.g., a probabilidade aumenta ou diminui com a `experiencia_piloto_anos`?)

## 4. Desafios e Lições

-   **Overfitting vs. Generalização:** Compara a performance da árvore única (provavelmente com *overfitting*) com a do Random Forest. Discute como o *bagging* ajuda a reduzir a variância e a melhorar a generalização.
-   **Viés-Variância Trade-off:** Pensa nos modelos em termos deste *trade-off*. Uma árvore tem alta variância. O Random Forest reduz a variância. O Boosting reduz o viés (sequencialmente).
-   **Explicabilidade do "Black Box":** Modelos ensemble são muitas vezes vistos como "caixas pretas". Discute como técnicas como *Feature Importance* e PDP nos permitem abrir essa caixa e ganhar confiança nas previsões.

## Produto Final

Um relatório que inclua:  
-   O treino e a avaliação dos modelos (Árvore, RF, XGBoost).  
-   Uma tabela comparativa de métricas de desempenho.  
-   Um gráfico de barras com o **ranking de importância das variáveis**.  
-   Um ou dois **gráficos de dependência parcial** com a sua interpretação.  
-   Uma conclusão sobre o poder dos modelos ensemble tanto para predição quanto para extração de conhecimento.  