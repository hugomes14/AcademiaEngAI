# Avaliação do Melhor Subconjunto

## Features selecionadas

- `sensor_07`
- `sensor_09`
- `sensor_14`
- `stress_operacional`

## Métricas

| modelo                            |   n_features |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:----------------------------------|-------------:|-----------:|------------:|---------:|-------:|----------:|
| LogisticRegression_todas_features |           18 |        0.9 |      0.6667 |   0.9655 | 0.7887 |    0.9584 |
| LogisticRegression_features_GA    |            4 |        0.9 |      0.6667 |   0.9655 | 0.7887 |    0.9672 |

## Matriz de Confusão

![Matriz de Confusão](matriz_confusao_ga.png)

## Relatório de classificação

```text
              precision    recall  f1-score   support

           0       0.99      0.88      0.93       121
           1       0.67      0.97      0.79        29

    accuracy                           0.90       150
   macro avg       0.83      0.92      0.86       150
weighted avg       0.93      0.90      0.91       150

```
