# Resumo de Explicabilidade

## Melhor modelo avaliado

- Modelo: `xgboost`

## Variáveis analisadas por PDP

- `horas_voo_desde_ultima_manutencao`
- `previsao_turbulencia`

## Leitura recomendada

- A importância de variáveis ajuda a identificar quais atributos pesam mais na decisão.
- O PDP ajuda a perceber se o aumento de uma variável tende a aumentar ou reduzir a probabilidade prevista de incidente.
- Como o dataset é desbalanceado, a interpretação deve ser feita em conjunto com métricas como Recall, F1 e PR-AUC.
