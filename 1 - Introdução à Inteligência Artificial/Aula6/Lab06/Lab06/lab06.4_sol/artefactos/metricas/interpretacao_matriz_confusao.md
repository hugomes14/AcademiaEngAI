# Interpretação da Matriz de Confusão

## Melhor modelo

Modelo: `xgboost`

## Valores

- TN — Verdadeiros negativos: 180
- FP — Falsos positivos: 0
- FN — Falsos negativos: 0
- TP — Verdadeiros positivos: 20

## Indicadores derivados

- FPR — Taxa de falsos positivos: 0.0000
- FNR — Taxa de falsos negativos: 0.0000
- Specificity — capacidade de detetar corretamente voos sem incidente: 1.0000
- Recall — capacidade de detetar corretamente incidentes: 1.0000

## Leitura no contexto do problema

Os falsos negativos são especialmente críticos: representam voos com incidente real que o
modelo classificou como seguros. Os falsos positivos podem gerar custos operacionais, mas
são geralmente menos graves do que ignorar um risco real.
