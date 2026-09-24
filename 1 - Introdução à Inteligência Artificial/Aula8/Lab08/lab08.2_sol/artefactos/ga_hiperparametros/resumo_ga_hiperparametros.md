# GA de Hiperparâmetros

## Melhores hiperparâmetros

```json
{
  "n_estimators": 90,
  "max_depth": null,
  "min_samples_split": 6,
  "max_features": "sqrt"
}
```

## Métricas no conjunto de teste

| modelo                          |   n_features |   accuracy |   precision |   recall |     f1 |   roc_auc |   fitness_cv_f1 |
|:--------------------------------|-------------:|-----------:|------------:|---------:|-------:|----------:|----------------:|
| RandomForest_GA_Hiperparametros |            4 |       0.96 |      0.8966 |   0.8966 | 0.8966 |    0.9883 |          0.7505 |

## Interpretação

O GA explorou combinações de hiperparâmetros do Random Forest. A fitness foi o F1-score médio em validação cruzada.
