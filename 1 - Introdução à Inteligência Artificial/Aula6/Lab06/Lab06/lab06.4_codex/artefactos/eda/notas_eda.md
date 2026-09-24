# Notas da Análise Exploratória

## Dataset

- Linhas: 1000
- Colunas: 6
- Variável-alvo: `incidente_reportado`

## Desbalanceamento

|   classe |   contagem |   percentagem |
|---------:|-----------:|--------------:|
|        0 |        900 |            90 |
|        1 |        100 |            10 |

Este dataset é desbalanceado, porque a classe de incidentes representa uma minoria.
Por isso, a Accuracy não deve ser usada isoladamente. Um modelo que prevê sempre
"sem incidente" pode ter uma Accuracy elevada, mas não tem utilidade operacional.

## Decisões para as etapas seguintes

- Usar `train_test_split` estratificado.
- Aplicar codificação ordinal à turbulência, porque existe ordem natural: Baixa < Média < Alta.
- Aplicar one-hot encoding ao tipo de missão, porque não existe ordem natural.
- Usar `class_weight` nos modelos que suportam este parâmetro.
- Avaliar com F1, Recall, ROC-AUC e PR-AUC.
