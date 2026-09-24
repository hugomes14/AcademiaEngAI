# Lab 06.1 - Regressão — Estimar a Duração de Voo

**Tema:** Regressão Linear e Regularizada

## 1. Objetivo

O objetivo deste exercício é construir e avaliar modelos de regressão capazes de estimar a **duração total de um voo** com base em variáveis da missão disponíveis antes da descolagem. Vamos comparar modelos lineares simples, múltiplos e regularizados para perceber os seus trade-offs.

## 2. Dados

Irás trabalhar com um dataset tabular (`voos_telemetria.csv`) que contém as seguintes variáveis por cada voo:
- `distancia_planeada` (km)
- `carga_util_kg` (kg)
- `altitude_media_m` (metros)
- `condicao_meteo` (categórica: `Bom`, `Moderado`, `Adverso`)
- `duracao_voo_min` (minutos) — a nossa **variável alvo**.

## 3. Tarefas

1.  **Análise Exploratória:**
    *   Visualiza a distribuição da variável alvo (`duracao_voo_min`).
    *   Analisa a correlação entre as variáveis numéricas e a duração do voo.
    *   Converte a variável `condicao_meteo` para um formato numérico (e.g., *one-hot encoding*).

2.  **Modelagem:**
    *   **Modelo Base:** Treina um modelo de **Regressão Linear Simples** usando apenas a `distancia_planeada`.
    *   **Modelo Completo:** Treina um modelo de **Regressão Linear Múltipla** com todas as variáveis preditoras.
    *   **Modelos Regularizados:** Treina modelos **Lasso** e **Ridge** para avaliar o impacto da regularização.
    *   **Não Linearidade:** Explora a **Regressão Polinomial** (grau 2) para capturar relações não lineares.

3.  **Avaliação e Interpretação:**
    *   Para cada modelo, calcula as seguintes métricas no conjunto de teste: **MAE**, **RMSE** e **R²**.
    *   Compara o desempenho dos modelos. Qual deles generaliza melhor?
    *   Interpreta os coeficientes do modelo de Regressão Múltipla. O que nos dizem sobre a importância de cada variável?
    *   Analisa os coeficientes do modelo Lasso. Alguma variável foi considerada irrelevante?

## 4. Desafios e Lições

-   **Escalonamento de Dados:** Avalia a importância de escalar as variáveis (*features*) antes de treinar os modelos, especialmente para Lasso e Ridge.
-   **Interpretação vs. Predição:** Reflete sobre o compromisso entre a simplicidade/interpretabilidade de um modelo linear e a potencial maior precisão de um modelo mais complexo.
-   ***Data Leakage*:** Garante que nenhuma informação do futuro (que não estaria disponível antes do voo) é usada para treinar o modelo.

## Produto Final

Um relatório comparativo (em formato `Markdown` ou outro) que apresente:  
-   A análise exploratória.  
-   O código de treino para cada modelo.  
-   Uma tabela a resumir as métricas de desempenho.  
-   Uma conclusão sobre qual o melhor modelo para a tarefa e uma interpretação dos seus resultados.  