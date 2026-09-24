# RELATÓRIO FINAL — Lab 06.2 Classificação: Prever o Risco de Incidentes em Voos

## 1. Introdução

Este relatório resume um pipeline de Machine Learning supervisionado para classificação binária. O objetivo é prever se um voo poderá ter um incidente de segurança (`incidente_reportado = 1`) com base em dados disponíveis antes da descolagem.

Como o problema envolve segurança, os erros não têm todos o mesmo custo. Um falso negativo significa classificar como seguro um voo que teve incidente, o que pode ser crítico. Por esse motivo, a análise valoriza métricas como Recall, F1, ROC-AUC e PR-AUC, e não apenas Accuracy.

## 2. Dados e Análise Exploratória

- Dataset usado: `data/voos_pre_voo.csv`.
- Variável-alvo: `incidente_reportado`.
- Variáveis numéricas: `idade_aeronave_anos, horas_voo_desde_ultima_manutencao, experiencia_piloto_anos`.
- Variáveis ordinais: `previsao_turbulencia`.
- Variáveis nominais: `tipo_missao`.
- Distribuição no treino: `{'0': 675, '1': 75}`.
- Distribuição no teste: `{'0': 225, '1': 25}`.

![Distribuição da variável-alvo](../eda/01_distribuicao_alvo.png)

A análise exploratória confirma o desbalanceamento: a classe de incidente é minoritária. Isto torna a Accuracy insuficiente, porque um modelo que prevê sempre 'sem incidente' pode parecer bom, mas falhar exatamente os casos relevantes.

## 3. Pipeline de Pré-processamento

O pipeline separou os dados em treino e teste com split estratificado, preservando a proporção das classes. A variável `previsao_turbulencia` foi tratada como ordinal, com a ordem Baixa < Média < Alta. A variável `tipo_missao` foi codificada com one-hot encoding, por não ter uma ordem natural. As variáveis numéricas e a variável ordinal codificada foram escalonadas com `StandardScaler`.

Para evitar data leakage, os transformadores foram ajustados apenas no conjunto de treino. O conjunto de teste recebeu apenas a transformação já aprendida no treino.

## 4. Modelos Treinados

Foram treinados os seguintes modelos:
- Regressão Logística
- Regressão Logística com `class_weight='balanced'`
- K-Nearest Neighbors
- SVM com kernel linear
- SVM com kernel linear e `class_weight='balanced'`
- SVM com kernel RBF
- SVM com kernel RBF e `class_weight='balanced'`
- Naïve Bayes Gaussiano
- KNN e SVM RBF sem escalonamento, como comparação pedagógica sobre sensibilidade à escala.

## 5. Resultados

| modelo                       | dados_usados   |   accuracy |   precision |   recall |   specificity |     f1 |   roc_auc |   pr_auc |   fp |   fn |   tp |
|:-----------------------------|:---------------|-----------:|------------:|---------:|--------------:|-------:|----------:|---------:|-----:|-----:|-----:|
| SVM_RBF_Balanced             | scaled         |      1     |      1      |     1    |        1      | 1      |    1      |   1      |    0 |    0 |   25 |
| Regressao_Logistica_Balanced | scaled         |      0.996 |      0.9615 |     1    |        0.9956 | 0.9804 |    0.9998 |   0.9985 |    1 |    0 |   25 |
| SVM_Linear                   | scaled         |      0.992 |      1      |     0.92 |        1      | 0.9583 |    1      |   1      |    0 |    2 |   23 |
| SVM_RBF                      | scaled         |      0.992 |      1      |     0.92 |        1      | 0.9583 |    1      |   1      |    0 |    2 |   23 |
| Naive_Bayes_Gaussiano        | scaled         |      0.984 |      0.8889 |     0.96 |        0.9867 | 0.9231 |    0.9988 |   0.9889 |    3 |    1 |   24 |
| SVM_Linear_Balanced          | scaled         |      0.984 |      0.92   |     0.92 |        0.9911 | 0.92   |    0.9977 |   0.9826 |    2 |    2 |   23 |
| Regressao_Logistica          | scaled         |      0.984 |      1      |     0.84 |        1      | 0.913  |    1      |   1      |    0 |    4 |   21 |
| KNN                          | scaled         |      0.976 |      1      |     0.76 |        1      | 0.8636 |    0.9994 |   0.9912 |    0 |    6 |   19 |
| KNN_Sem_Escala               | unscaled       |      0.952 |      1      |     0.52 |        1      | 0.6842 |    0.852  |   0.7194 |    0 |   12 |   13 |
| SVM_RBF_Sem_Escala           | unscaled       |      0.944 |      1      |     0.44 |        1      | 0.6111 |    0.8658 |   0.7726 |    0 |   14 |   11 |


O melhor modelo pelo critério principal **F1-Score** foi **SVM_RBF_Balanced**. O F1 obtido foi **1.0000**, com ROC-AUC de **1.0000**.

A métrica F1 foi escolhida porque equilibra Precision e Recall. Numa tarefa de risco de incidentes, Recall é especialmente importante, porque reduz falsos negativos. No entanto, Precision também importa, pois demasiados falsos positivos podem gerar alarmes operacionais desnecessários.

## 6. Matriz de Confusão

![Matriz de Confusão](../graficos/matriz_confusao_melhor_modelo.png)

- TN: 225 — voos sem incidente classificados como sem incidente.
- FP: 0 — voos sem incidente classificados como incidentes.
- FN: 0 — incidentes reais classificados como sem incidente.
- TP: 25 — incidentes reais classificados como incidentes.
- Recall: 1.0000.
- Specificity: 1.0000.

Os falsos negativos merecem atenção especial. Em contexto de segurança, um falso negativo pode significar que uma missão avança sem inspeção adicional apesar de existir risco.

## 7. Curvas ROC e Precision-Recall

![Curvas ROC](../graficos/curvas_roc_modelos.png)

![Curvas Precision-Recall](../graficos/curvas_precision_recall_modelos.png)

A curva ROC permite comparar a separabilidade global dos modelos. A curva Precision-Recall é particularmente relevante neste dataset, porque a classe positiva é rara. Quando existe desbalanceamento, PR-AUC pode ser mais informativa do que ROC-AUC.

## 8. Conclusões e Recomendações

O melhor modelo deve ser escolhido não só pela métrica agregada, mas também pelo custo operacional dos erros. Se o objetivo principal for minimizar incidentes não detetados, recomenda-se otimizar o threshold para aumentar Recall, mesmo que isso reduza Precision.

Recomendações para continuação:
- Testar thresholds diferentes de 0.5 para ajustar o equilíbrio Precision/Recall.
- Comparar `class_weight='balanced'` com técnicas de reamostragem, como oversampling ou undersampling.
- Fazer tuning de hiperparâmetros com validação cruzada estratificada.
- Analisar outliers e casos mal classificados para perceber padrões operacionais.
- Considerar novas variáveis preditoras disponíveis antes da descolagem, sem introduzir data leakage.
- Rever o impacto de falsos positivos e falsos negativos com especialistas do domínio.

## 9. Referências

- Documentação oficial do scikit-learn sobre métricas de classificação.
- Documentação oficial do scikit-learn sobre pipelines e pré-processamento.
- Boas práticas de avaliação em datasets desbalanceados.
