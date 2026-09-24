# RELATÓRIO FINAL — Lab 06.4: Modelos Ensemble

## 1. Introdução

Este relatório apresenta a resolução do laboratório de métodos ensemble para previsão de
incidentes em voos. O problema é uma tarefa de **classificação binária**:

- Classe 0: voo sem incidente reportado.
- Classe 1: voo com incidente reportado.

O objetivo é comparar uma Árvore de Decisão, Random Forest e modelos de boosting,
incluindo XGBoost quando disponível, e usar técnicas de explicabilidade para compreender
as previsões.

## 2. Dataset

O dataset utilizado foi `voos_pre_voo.csv`.

### Distribuição da variável-alvo

|   classe |   contagem |   percentagem |
|---------:|-----------:|--------------:|
|        0 |        900 |            90 |
|        1 |        100 |            10 |

![Distribuição do alvo](../figuras/eda_distribuicao_alvo.png)

O dataset está desbalanceado. A classe positiva é minoritária, por isso a Accuracy deve
ser interpretada com cuidado.

## 3. Análise Exploratória

Foram analisadas:

- variáveis numéricas por classe;
- variáveis categóricas por classe;
- correlações entre variáveis numéricas e alvo;
- desbalanceamento da variável-alvo.

![Correlação](../figuras/eda_heatmap_correlacoes.png)

## 4. Pré-processamento

O pipeline aplicou:

- split treino/teste estratificado;
- `StandardScaler` nas variáveis numéricas;
- encoding ordinal em `previsao_turbulencia`, com ordem Baixa < Média < Alta;
- one-hot encoding em `tipo_missao`;
- `fit` apenas no treino, para evitar data leakage.

## 5. Modelos

Foram treinados:

- `DecisionTreeClassifier`, como baseline e demonstração de potencial overfitting;
- `RandomForestClassifier`, como bagging;
- `XGBClassifier`, como boosting, se disponível;
- `GradientBoostingClassifier`, como alternativa se o XGBoost não estiver disponível;
- `CatBoostClassifier`, se disponível.

### Random Forest OOB

OOB score: 0.993750

### Tempos de treino

| modelo                     |   tempo_treino_segundos |
|:---------------------------|------------------------:|
| decision_tree              |               0.0184868 |
| random_forest              |               0.261339  |
| gradient_boosting_fallback |               0.245468  |

## 6. Resultados

| modelo                     |   accuracy |   precision |   recall |   specificity |     f1 |   roc_auc |   pr_auc |   tn |   fp |   fn |   tp |
|:---------------------------|-----------:|------------:|---------:|--------------:|-------:|----------:|---------:|-----:|-----:|-----:|-----:|
| gradient_boosting_fallback |       1    |        1    |     1    |        1      | 1      |    1      |   1      |  180 |    0 |    0 |   20 |
| decision_tree              |       0.99 |        0.95 |     0.95 |        0.9944 | 0.95   |    0.9722 |   0.9075 |  179 |    1 |    1 |   19 |
| random_forest              |       0.99 |        1    |     0.9  |        1      | 0.9474 |    1      |   1      |  180 |    0 |    2 |   18 |

### Melhor modelo

Pelo critério principal **F1-Score**, o melhor modelo foi:

- Modelo: `gradient_boosting_fallback`
- F1: 1.0000
- ROC-AUC: 1.0000
- Recall: 1.0000
- Precision: 1.0000

## 7. Matriz de Confusão

![Matriz de Confusão](../figuras/matriz_confusao_melhor_modelo.png)

Os falsos negativos são especialmente críticos porque representam incidentes reais que
não foram assinalados pelo modelo. Os falsos positivos geram custos operacionais, mas,
num cenário de segurança, podem ser mais aceitáveis do que falhar incidentes.

## 8. Curvas ROC e Precision-Recall

![Curvas ROC](../figuras/curvas_roc_modelos.png)

![Curvas Precision-Recall](../figuras/curvas_precision_recall_modelos.png)

A curva Precision-Recall é particularmente útil neste problema, porque a classe positiva
é rara.

## 9. Explicabilidade

### Árvore de Decisão

![Árvore de Decisão](../figuras/arvore_decisao_primeiros_niveis.png)

A árvore isolada permite interpretação visual, mas pode apresentar alta variância e
sobreajuste quando não tem limite de profundidade.

### Importância de Variáveis


### decision_tree

| modelo        | feature                           |   importancia |
|:--------------|:----------------------------------|--------------:|
| decision_tree | horas_voo_desde_ultima_manutencao |        0.6477 |
| decision_tree | idade_aeronave_anos               |        0.1844 |
| decision_tree | experiencia_piloto_anos           |        0.1016 |
| decision_tree | previsao_turbulencia              |        0.0664 |
| decision_tree | tipo_missao_Carga                 |        0      |
| decision_tree | tipo_missao_Transporte            |        0      |
| decision_tree | tipo_missao_Vigilância            |        0      |

### gradient_boosting_fallback

| modelo                     | feature                           |   importancia |
|:---------------------------|:----------------------------------|--------------:|
| gradient_boosting_fallback | horas_voo_desde_ultima_manutencao |        0.7755 |
| gradient_boosting_fallback | previsao_turbulencia              |        0.1061 |
| gradient_boosting_fallback | idade_aeronave_anos               |        0.0811 |
| gradient_boosting_fallback | experiencia_piloto_anos           |        0.037  |
| gradient_boosting_fallback | tipo_missao_Vigilância            |        0.0003 |
| gradient_boosting_fallback | tipo_missao_Carga                 |        0      |
| gradient_boosting_fallback | tipo_missao_Transporte            |        0      |

### random_forest

| modelo        | feature                           |   importancia |
|:--------------|:----------------------------------|--------------:|
| random_forest | horas_voo_desde_ultima_manutencao |        0.348  |
| random_forest | previsao_turbulencia              |        0.2478 |
| random_forest | experiencia_piloto_anos           |        0.1827 |
| random_forest | idade_aeronave_anos               |        0.1668 |
| random_forest | tipo_missao_Vigilância            |        0.0263 |
| random_forest | tipo_missao_Transporte            |        0.0257 |
| random_forest | tipo_missao_Carga                 |        0.0027 |


### Partial Dependence Plots

![Partial Dependence Plots](../figuras/pdp_variaveis_importantes.png)

Os PDP ajudam a perceber como as variáveis mais relevantes influenciam a probabilidade
prevista de incidente.

## 10. Conclusões

Os modelos ensemble permitem melhorar a robustez face a uma árvore isolada:

- Random Forest reduz variância ao combinar várias árvores.
- Boosting tenta reduzir erro sequencialmente, ao corrigir falhas de modelos anteriores.
- As métricas F1, Recall, ROC-AUC e PR-AUC são mais adequadas do que Accuracy para este
  cenário desbalanceado.

## 11. Recomendações

- Afinar hiperparâmetros com `GridSearchCV` ou `RandomizedSearchCV`.
- Testar thresholds diferentes de 0.5 para aumentar Recall ou Precision conforme o custo operacional.
- Avaliar reamostragem ou `class_weight` com mais detalhe.
- Usar SHAP, quando disponível, para explicabilidade local.
- Comparar os resultados com os modelos simples do Lab 06.2.
