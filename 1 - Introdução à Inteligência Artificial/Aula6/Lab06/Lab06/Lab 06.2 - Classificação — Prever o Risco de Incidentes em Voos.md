# Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos

**Tema:** Classificação Binária e Avaliação de Modelos

## 1. Objetivo

Neste laboratório, o desafio é desenvolver um modelo de classificação capaz de prever a probabilidade de um **incidente de segurança** (`risco = 1`) ocorrer durante um voo, utilizando dados disponíveis *antes* da descolagem. O foco será na avaliação e comparação de diferentes algoritmos e na interpretação das suas previsões.

## 2. Dados

Utilizarás um dataset tabular (`voos_pre_voo.csv`) com as seguintes características:
- `idade_aeronave_anos`
- `horas_voo_desde_ultima_manutencao`
- `previsao_turbulencia` (Baixa, Média, Alta)
- `tipo_missao` (Vigilância, Carga, Transporte)
- `experiencia_piloto_anos`
- `incidente_reportado` (0 ou 1) — a nossa **variável alvo**.

Este é um dataset **desbalanceado**: a maioria dos voos não tem incidentes.

## 3. Tarefas

1.  **Análise e Pré-processamento:**
    *   Verifica o desbalanceamento da variável alvo. Quantos voos pertencem a cada classe?
    *   Analisa a relação entre as variáveis preditoras e a ocorrência de incidentes.
    *   Converte as variáveis categóricas (`previsao_turbulencia`, `tipo_missao`) para formato numérico.
    *   Escalona as variáveis numéricas, pois modelos como KNN e SVM são sensíveis à escala.

2.  **Modelagem:**
    *   Treina e avalia os seguintes modelos de classificação:
        *   **Regressão Logística** (como *baseline*)
        *   **K-Nearest Neighbors (KNN)**
        *   **Support Vector Machine (SVM)** com kernel linear
        *   **Kernel SVM** (com kernel RBF)
        *   **Naïve Bayes** (Gaussiano)

3.  **Avaliação Detalhada:**
    *   Para cada modelo, calcula as métricas: **Accuracy**, **Precision**, **Recall**, **F1-Score** e **ROC-AUC**.
    *   Gera a **Matriz de Confusão** para o melhor modelo (baseado no F1-Score ou ROC-AUC). Discute os Falsos Positivos e Falsos Negativos no contexto do problema.
    *   Plota a **Curva ROC** para todos os modelos no mesmo gráfico para comparar visualmente o seu desempenho.

## 4. Desafios e Lições

-   **Gestão de Desbalanceamento:** Como o desbalanceamento afeta as métricas? Discute a relevância da *Accuracy* neste cenário e por que *Recall* e *F1-Score* são mais importantes. (Opcional: experimenta técnicas como `class_weight='balanced'` ou over-sampling/under-sampling).
-   **Escolha do Threshold:** A Regressão Logística e o SVM fornecem probabilidades. Como a escolha de um *threshold* de decisão (diferente do padrão 0.5) pode otimizar o *trade-off* entre Precision e Recall?
-   **Sensibilidade à Escala:** Compara o desempenho do KNN ou SVM com e sem escalonamento de dados para demonstrar a sua importância.

## Produto Final

Um relatório (em formato `Markdown` ou outro) contendo:  
-   A análise do desbalanceamento dos dados.  
-   O pipeline de pré-processamento.  
-   O código de treino e avaliação dos modelos.  
-   Uma tabela comparativa das métricas.  
-   A Matriz de Confusão e a Curva ROC comentadas, com uma justificação para a escolha do "melhor" modelo para esta tarefa específica.  