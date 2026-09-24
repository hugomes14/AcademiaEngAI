# Aula 6 - Aprendizagem supervisionada: regressão, classificação, séries temporais, ensembles e deep learning

## Registo

- Data do registo: 2026-07-24
- Diretório de origem: `Aula6/`
- Módulo: `Módulo 06`
- Laboratórios: `Lab 06.1` a `Lab 06.5`

## Resumo

Esta aula aprofunda a aprendizagem supervisionada através de cinco problemas aplicados ao contexto de operações de voo. Os laboratórios evoluem da regressão linear e classificação binária para previsão de séries temporais, métodos de conjunto e uma introdução prática a redes neuronais com PyTorch.

Em todos os problemas, as decisões importantes incluem separar corretamente treino e teste, evitar data leakage, selecionar métricas adequadas e interpretar os erros no contexto operacional.

## Problemas de cada laboratório

### Lab 06.1 - Regressão: estimar a duração de voo

**Problema:** prever um valor contínuo, `duracao_voo_min`, antes da descolagem.

O modelo recebe a distância planeada, a carga útil, a altitude média e a condição meteorológica. O objetivo é estimar quantos minutos irá durar o voo. São comparados modelos de regressão linear simples e múltipla, Ridge, Lasso e regressão polinomial.

O laboratório aborda a distribuição do alvo, outliers, encoding da meteorologia, escalonamento, RMSE/MAE/R², previsão versus valor real e análise de resíduos. O principal risco operacional é a subestimação da duração, que pode afetar autonomia, combustível, equipas e janelas de operação.

### Lab 06.2 - Classificação: prever o risco de incidentes em voos

**Problema:** decidir se um voo tem risco de incidente, `incidente_reportado = 1`, a partir de informação disponível antes da descolagem.

As entradas incluem idade da aeronave, horas desde a última manutenção, previsão de turbulência, tipo de missão e experiência do piloto. Como incidentes são raros, o dataset é desbalanceado: prever sempre a classe sem incidente pode produzir uma Accuracy alta, mas não resolve o problema.

São comparadas Regressão Logística, KNN, SVM linear, SVM com kernel RBF e Naive Bayes. O foco está em Precision, Recall, F1, ROC-AUC, curvas Precision-Recall e matriz de confusão. Os falsos negativos são especialmente importantes, pois representam incidentes reais que o sistema não sinalizou.

### Lab 06.3 - Séries temporais: prever procura e consumo

**Problema:** prever valores futuros respeitando a ordem cronológica dos dados.

O laboratório tem duas séries: a voltagem de uma bateria medida minuto a minuto e o número diário de missões. Para a voltagem, o objetivo é antecipar a evolução de curto prazo e possíveis quedas ou picos. Para as missões, o objetivo é apoiar o planeamento de recursos nas semanas seguintes.

São analisadas frequência, lacunas, tendência, sazonalidade, estacionariedade, ACF e PACF. O treino e o teste usam um split temporal, nunca aleatório. O baseline de persistência é comparado com regressão linear, Ridge, Random Forest e suavização exponencial; ARIMA/SARIMA e Prophet constituem extensões adequadas. As métricas incluem MAE, MSE, RMSE e MAPE/sMAPE.

### Lab 06.4 - Modelos ensemble: maximizar desempenho e explicar previsões

**Problema:** melhorar a previsão de `incidente_reportado` através de ensembles e explicar quais as variáveis mais influentes.

É reutilizado o problema de classificação do Lab 06.2. Uma Árvore de Decisão serve como baseline, enquanto Random Forest aplica bagging para reduzir a variância e XGBoost ou Gradient Boosting aplica boosting para corrigir erros de forma sequencial. CatBoost surge como extensão opcional.

Além de comparar F1, ROC-AUC, Recall e métricas de desbalanceamento, o laboratório analisa o OOB score do Random Forest. A explicabilidade inclui importância de variáveis, importância por permutação e Partial Dependence Plots. Estes resultados apoiam a compreensão do modelo, mas não demonstram causalidade.

### Lab 06.5 - Deep learning: regressão com uma rede neural de um neurónio

**Problema:** estimar `duracao_voo_min` com uma rede neural PyTorch e demonstrar a equivalência matemática entre um neurónio linear e a regressão linear múltipla.

O dataset é o mesmo do Lab 06.1. Após one-hot encoding de `condicao_meteo` e padronização das features, os dados são divididos em treino, validação e teste. O modelo PyTorch possui uma única camada `nn.Linear`, sem função de ativação, e aprende com MSE e um otimizador como Adam ou SGD.

O laboratório introduz tensors, autograd, forward pass, backward pass, loop de treino, loss de validação e early stopping. A regressão linear do scikit-learn usa o mesmo pré-processamento para permitir a comparação de MAE, MSE, RMSE, R², pesos e bias. Pequenas diferenças entre os resultados podem ocorrer por inicialização, learning rate, número de épocas e convergência.

## Conceitos-chave da aula

- **Aprendizagem supervisionada:** aprendizagem a partir de exemplos com uma variável-alvo conhecida.
- **Regressão:** previsão de um valor contínuo, como a duração de voo.
- **Classificação binária:** previsão de uma classe, como incidente ou sem incidente.
- **Desbalanceamento:** situação em que uma classe é muito menos frequente, o que exige métricas além de Accuracy.
- **Data leakage:** uso indevido de informação do teste, do futuro ou do alvo durante o treino.
- **Validação temporal:** separação cronológica obrigatória para previsão de séries temporais.
- **Ensemble:** combinação de vários modelos para reduzir variância, viés ou ambos.
- **Explicabilidade:** técnicas que ajudam a compreender a contribuição das features, sem inferir causalidade.
- **Rede neural linear:** um neurónio sem ativação calcula uma combinação linear das features e um bias.

## Materiais no repositório

- `Aula6/Módulo 06/Módulo 06/Módulo 06 - Tipos de aprendizagem - aprendizagem supervisionada.pdf`
- `Aula6/Lab06/Lab06/Lab 06.1 - Regressão — Estimar a Duração de Voo.md`
- `Aula6/Lab06/Lab06/Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos.md`
- `Aula6/Lab06/Lab06/Lab 06.3 - Séries Temporais — Previsão de procura e consumo.md`
- `Aula6/Lab06/Lab06/Lab 06.4 - Modelos Ensemble — Maximizar Desempenho e Explicar Previsões.md`
- `Aula6/Lab06/Lab06/Lab 06.5 - Aprendizagem Profunda (Deep Learning) — Regressão com Rede Neural (1 Neurónio).md`

## Notas de estudo

- A métrica deve corresponder ao problema: RMSE/MAE para regressão; F1, Recall, ROC-AUC e PR-AUC para classificação desbalanceada.
- Um modelo só deve usar dados que estariam disponíveis no momento real da decisão.
- O baseline é essencial: uma previsão útil deve superar uma estratégia simples, como persistência numa série temporal.
- Melhor desempenho não dispensa análise de erros, limitações dos dados e validação futura.
- As importâncias de variáveis e os coeficientes mostram associações no modelo, não relações de causa e efeito.
