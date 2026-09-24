# RELATÓRIO FINAL — Lab 08.2: Algoritmos Genéticos

## 1. Introdução

Este laboratório demonstra o uso de Algoritmos Genéticos em dois tipos de problemas:

1. **Seleção de features** para classificação binária.
2. **Otimização de hiperparâmetros** de um modelo de Machine Learning.
3. **Otimização multiobjetivo de rotas**, com análise da Frente de Pareto.

A ideia central é trabalhar com uma população de soluções candidatas. Cada solução é avaliada por uma função de fitness e a população evolui através de seleção, cruzamento e mutação.

## 2. Dados e EDA

O projeto é autocontido. Se não existir um dataset real, o script inicial gera um dataset sintético de voos com features de sensores e uma variável-alvo binária: `incidente_reportado`.

Artefactos principais:

- `artefactos/eda/distribuicao_target.png`
- `artefactos/eda/histogramas_features.png`
- `artefactos/eda/heatmap_correlacoes.png`

![Distribuição do target](../eda/distribuicao_target.png)

![Heatmap de correlações](../eda/heatmap_correlacoes.png)

## 3. Pipeline de Pré-processamento

O pipeline separa treino e teste com estratificação. Depois aplica `StandardScaler`, ajustado apenas no treino, para evitar *data leakage*.

A estratificação preserva a proporção de classes no treino e no teste.

## 4. Algoritmo Genético para Seleção de Features

### Representação do indivíduo

Cada indivíduo é uma lista binária:

- `1`: a feature entra no modelo.
- `0`: a feature fica fora do modelo.

### Função de fitness

A fitness usa o F1-score médio em validação cruzada, com uma pequena penalização por complexidade. Isto incentiva modelos mais simples, sem perder desempenho.

### Features selecionadas

- `sensor_07`
- `sensor_09`
- `sensor_14`
- `stress_operacional`

### Convergência

![Convergência do GA](../ga_features/convergencia_ga.png)

![Tamanho dos subconjuntos](../ga_features/tamanho_subconjuntos_ga.png)

## 5. Avaliação do Melhor Subconjunto

| modelo                            |   n_features |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:----------------------------------|-------------:|-----------:|------------:|---------:|-------:|----------:|
| LogisticRegression_todas_features |           18 |        0.9 |      0.6667 |   0.9655 | 0.7887 |    0.9584 |
| LogisticRegression_features_GA    |            4 |        0.9 |      0.6667 |   0.9655 | 0.7887 |    0.9672 |

![Matriz de Confusão](../avaliacao/matriz_confusao_ga.png)

### Interpretação

O modelo com features selecionadas deve ser comparado com o modelo base que usa todas as features. Mesmo que o desempenho fique ligeiramente abaixo, o subconjunto pode ser útil por reduzir complexidade, custo de recolha de dados e risco de sobreajuste.

## 6. Otimização de Hiperparâmetros

Foi implementado um AG simples para procurar hiperparâmetros de um `RandomForestClassifier`.

| modelo                          |   n_features |   accuracy |   precision |   recall |     f1 |   roc_auc |   fitness_cv_f1 |
|:--------------------------------|-------------:|-----------:|------------:|---------:|-------:|----------:|----------------:|
| RandomForest_GA_Hiperparametros |            4 |       0.96 |      0.8966 |   0.8966 | 0.8966 |    0.9883 |          0.7505 |

![Convergência dos hiperparâmetros](../ga_hiperparametros/convergencia_ga_hiperparametros.png)

## 7. Otimização Multiobjetivo de Rotas

Nesta etapa, cada indivíduo representa uma rota com waypoints intermédios. O algoritmo procura minimizar dois objetivos:

- tempo de voo;
- consumo de combustível.

O resultado é uma Frente de Pareto, ou seja, um conjunto de soluções não dominadas.

| rota                              | waypoints_intermedios   |   tempo_voo |   consumo_combustivel |
|:----------------------------------|:------------------------|------------:|----------------------:|
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |

![Frente de Pareto](../nsga2_rotas/frente_pareto_rotas.png)

## 8. Conclusões

Os Algoritmos Genéticos são úteis quando o espaço de procura é grande, discreto, irregular ou difícil de otimizar com métodos tradicionais.

Principais aprendizagens:

- A função de fitness define o comportamento do algoritmo.
- A mutação ajuda a manter diversidade.
- O elitismo preserva boas soluções.
- A validação cruzada reduz o risco de escolher features que apenas funcionam por acaso no treino.
- A Frente de Pareto é adequada quando existem objetivos concorrentes.

## 9. Próximos passos

- Testar uma implementação com `deap`.
- Aumentar o número de gerações e a população.
- Comparar a seleção por GA com métodos como `SelectKBest`, Lasso ou Random Forest importance.
- Acrescentar custo de sensores à função de fitness.
- Usar dados reais de telemetria e manutenção.
