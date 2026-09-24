# Notas da EDA

## Dataset
- Ficheiro: `dados_voo_ga.csv`
- Linhas: 600
- Colunas: 19
- Variável-alvo: `incidente_reportado`

## Distribuição da variável-alvo

|   incidente_reportado |   contagem |   percentagem |
|----------------------:|-----------:|--------------:|
|                     0 |        484 |         80.67 |
|                     1 |        116 |         19.33 |

## Observações pedagógicas
- O dataset simula uma tarefa de classificação binária.
- A seleção de features por Algoritmo Genético é útil porque existe um conjunto relativamente grande de variáveis.
- Algumas features podem ser redundantes, ruidosas ou pouco relevantes.
- O AG vai tentar encontrar um subconjunto com bom desempenho preditivo sem usar todas as features.
- A avaliação da fitness deve usar validação cruzada para reduzir o risco de sobreajuste ao treino.
