# Discussão das Métricas

## Melhor modelo

O melhor modelo pelo critério principal **F1-Score** foi:

- Modelo: `gradient_boosting_fallback`
- F1: 1.0000
- ROC-AUC: 1.0000
- Recall: 1.0000
- Precision: 1.0000

## Por que a Accuracy não chega?

Como a classe positiva representa uma minoria, um modelo que prevê sempre "sem incidente"
pode alcançar uma Accuracy elevada. No entanto, esse modelo falha precisamente no que interessa:
detetar incidentes.

## Custo de erros

- **Falso Negativo (FN):** o modelo prevê ausência de incidente, mas ocorre um incidente. É o erro mais crítico em segurança.
- **Falso Positivo (FP):** o modelo prevê risco, mas não ocorre incidente. Pode gerar inspeções adicionais e custos operacionais.

## Métrica principal

O F1-Score foi escolhido porque equilibra Precision e Recall. Num contexto de segurança,
também se deve observar o Recall, porque falhar incidentes reais pode ter custo elevado.
