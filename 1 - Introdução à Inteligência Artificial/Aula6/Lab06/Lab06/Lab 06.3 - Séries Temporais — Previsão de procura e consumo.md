# Lab 06.3 - Séries Temporais — Previsão de procura e consumo

**Tema:** Análise e Previsão de Séries Temporais

## 1. Objetivo

Este laboratório foca-se em modelar e prever duas séries temporais distintas:
1.  O **consumo de voltagem** de uma bateria, medido minuto a minuto, para antecipar picos ou quedas.
2.  O **número de missões de voo diárias**, para otimizar o planeamento de recursos.

Iremos explorar desde modelos clássicos até abordagens mais modernas como o Prophet.

## 2. Dados

Trabalharás com dois datasets:
1.  `voltagem_bateria.csv`:
    *   `timestamp` (a cada minuto)
    *   `voltagem` (V) — **série alvo 1**.
2.  `missoes_diarias.csv`:
    *   `data` (diária)
    *   `num_missoes` — **série alvo 2**.

## 3. Tarefas

### Parte A: Previsão de Voltagem (Série de Alta Frequência)

1.  **Análise Exploratória:**
    *   Visualiza a série temporal de voltagem. Identifica tendências, sazonalidades (diária, semanal?) e anomalias.
    *   Verifica a **estacionariedade** da série usando testes estatísticos (e.g., Augmented Dickey-Fuller). Se não for estacionária, aplica diferenciação.
    *   Plota as funções de Autocorrelação (ACF) e Autocorrelação Parcial (PACF) para ajudar a identificar os parâmetros (p, d, q) de um modelo ARIMA.

2.  **Modelagem Clássica:**
    *   Treina um modelo **ARIMA** (ou **SARIMA** se houver sazonalidade clara) para prever a voltagem nos próximos 60 minutos.
    *   Experimenta um modelo de **Suavização Exponencial** (e.g., Holt-Winters) e compara os resultados.

### Parte B: Previsão de Missões Diárias (Série Sazonal)

1.  **Análise Exploratória:**
    *   Visualiza a série de missões diárias. Procura por tendências de longo prazo e sazonalidades (semanal, mensal, anual).
    *   Decompõe a série nos seus componentes de tendência, sazonalidade e resíduo.

2.  **Modelagem com Prophet e LSTM:**
    *   Treina um modelo **Prophet** (do Facebook) para prever o número de missões para as próximas 4 semanas. Tira partido da sua capacidade de modelar múltiplos padrões sazonais e feriados.
    *   (Opcional Avançado) Treina um modelo **LSTM (Long Short-Term Memory)** simples. Compara a sua complexidade e desempenho com o Prophet.

3.  **Avaliação:**
    *   Para ambos os cenários, avalia as previsões usando as métricas **MAE**, **RMSE** e **MAPE** (ou sMAPE).
    *   Visualiza as previsões sobrepostas aos dados reais, incluindo os **intervalos de confiança/previsão**.

## 4. Desafios e Lições

-   **Estacionariedade:** Compreende por que a estacionariedade é uma premissa fundamental para modelos como o ARIMA e como as transformações (diferenciação, log) ajudam a alcançá-la.
-   **Validação Temporal:** A validação cruzada tradicional não funciona para séries temporais. Utiliza uma abordagem de *walk-forward validation* ou um simples *train-test split* cronológico.
-   **Escolha do Modelo:** Reflete sobre quando usar um modelo clássico como ARIMA vs. uma ferramenta mais automatizada como o Prophet. Quais são os prós e contras de cada um?

## Produto Final

Um relatório que documente:  
-   A análise e pré-processamento de cada série temporal.  
-   O código de treino para os modelos aplicados (ARIMA/SARIMA, Prophet, etc.).  
-   Gráficos que mostram as previsões, os valores reais e os intervalos de confiança.  
-   Uma discussão sobre os resultados e as características de cada série que influenciaram a escolha do modelo.  