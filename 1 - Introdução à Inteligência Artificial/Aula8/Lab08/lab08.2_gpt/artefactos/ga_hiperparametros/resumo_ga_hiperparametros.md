# GA de Hiperparâmetros

## Melhores hiperparâmetros

```json
{
  "n_estimators": 60,
  "max_depth": null,
  "min_samples_split": 2,
  "max_features": "sqrt"
}
```

## Métricas no conjunto de teste

| modelo                          |   n_features |   accuracy |   precision |   recall |   f1 |   roc_auc |   fitness_cv_f1 |
|:--------------------------------|-------------:|-----------:|------------:|---------:|-----:|----------:|----------------:|
| RandomForest_GA_Hiperparametros |            4 |       0.96 |       0.871 |    0.931 |  0.9 |    0.9842 |          0.7371 |

## Interpretação

O GA explorou combinações de hiperparâmetros do Random Forest. A fitness foi o F1-score médio em validação cruzada.
