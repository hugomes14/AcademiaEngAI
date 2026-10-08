# Classificador não linear e atualização da app — 2026-10-06

A pedido do utilizador, o classificador ativo passou da regressão logística
para **Extra Trees**, em `outputs/classification/posture_nonlinear_03/`.
O RTMPose mantém os mesmos pesos; os modelos anteriores foram preservados.
A regressão anterior era logística, não regressão linear; tinha uma fronteira
linear nos descritores geométricos.

## Dados e comparação

Mesmas 200 janelas de um segundo, 136 boas e 64 más, de oito gravações em seis
grupos. Mesmas medidas e mesmas regras de exclusão dos trechos. Não foram
reexecutadas as poses nem usados os frames do vídeo de teste para escolher o
modelo. Atribuições por origem conservam os tutoriais relacionados no mesmo
lado dos splits.

Compararam-se quatro candidatos fixos, sem pesquisa de hiperparâmetros:
regressão logística L2 (0,1), Random Forest e Extra Trees (200 árvores,
profundidade máxima 6, mínimo de 3 amostras por folha, 70% de features por
split, seed 42) e SVM RBF (C=1, gamma=scale, sem calibração de probabilidades).
A imputação e normalização são ajustadas ao treino de cada fold. Os pesos
continuam equilibrados por classe e grupo.

Validação leave-one-group-out usada para escolher a família, com acurácia
equilibrada como critério e macro F1 para desempate:

| Candidato | Acurácia | Acurácia equilibrada | Macro F1 |
| --- | ---: | ---: | ---: |
| logistic | 52.00% | 51.47% | 0.500 |
| random_forest | 54.00% | 49.22% | 0.491 |
| extra_trees | 58.00% | 52.16% | 0.521 |
| svm_rbf | 51.00% | 49.91% | 0.487 |

Extra Trees venceu esta comparação. Os resultados do vencedor são **scores
de seleção**, não uma avaliação independente da escolha.

## Validação da escolha

Validação aninhada: em cada um dos seis folds externos, o grupo externo fica
fora de toda a seleção. Uma nova validação por grupo dentro dos dados de treino
escolhe a família; essa família é ajustada apenas aos grupos de treino e avaliada
no grupo externo. O protocolo pode escolher famílias diferentes em folds
diferentes; avalia o procedimento de seleção, não um único modelo final fixo.

Resultado externo: acurácia **52.00%**, acurácia equilibrada
**48.16%**, macro F1 **0.479**.
Não existe melhoria fiável demonstrada. Trocar o algoritmo não resolveu as
limitações de apenas oito gravações, projeções 2D, janelas correlacionadas e
rótulos de demonstrações que abordam aspetos diferentes da corrida.

## Modelo e integração

O modelo final Extra Trees usa todas as fontes rotuladas, após a seleção.
É exportado em JSON v2 com imputação, normalização e estrutura das árvores.
A inferência não carrega pickle/joblib nem precisa de scikit-learn. O exportador
verifica que os scores coincidem com o estimador original antes de guardar.
Modelos JSON v1 de regressão logística continuam suportados.

O manifesto `outputs/classification/active_classifier.json` seleciona a versão
ativa. O worker fixa o caminho do modelo quando a análise começa e regista
algoritmo, versão e SHA-256 no seu estado. Não sobrescreve versões antigas nem
altera resultados já concluídos. Sem manifesto, existe fallback para o primeiro
modelo. A app continua a assinalar que a classificação é experimental.

## Reprodução

Na raiz PosturaAI:

```bash
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python -m scripts.compare_classifiers --source-run outputs/classification/posture_baseline_02 --run-dir outputs/classification/posture_nonlinear_03
.venv/bin/python -m scripts.classify_video --model outputs/classification/posture_nonlinear_03/classifier.json --poses outputs/classification/posture_baseline_02/test/4116077361ff1b20/pose.poses.jsonl --pose-video outputs/classification/posture_baseline_02/test/4116077361ff1b20/pose.mp4 --output outputs/classification/posture_nonlinear_03/test/classified.mp4
```

A comparação não ativa automaticamente o vencedor. Nesta execução, a ativação
foi explícita, a pedido do utilizador, e está registada em `activation.json`.
Os parâmetros, folds internos/externos e métricas estão em `comparison_report.json`.
O vídeo de teste continua sem rótulos reais de postura; a inferência visual não
fornece métricas de acertos.

## Referências

- [Extra Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
- [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
- [Validação aninhada e viés de seleção](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Verificação de inferência

O vídeo de ritmos foi processado com o modelo novo e convertido para H.264:
248 frames, 171 com previsão, 77 sem previsão; entre as previsões apresentadas,
76 frames de má postura, 65 de boa e 30 inconclusivos. Os 248 frames do resultado
foram descodificados com sucesso. Não há rótulos reais para medir acertos.

O resultado completo está na app em `/postura-ai?job=df7f706d423c4d28b88d126d2cc3c41e`.

Foi também enviado um novo vídeo de dois segundos pelo upload real da app.
O job registou `extra_trees` e o hash do modelo ativo, acompanhou 50 frames,
produziu 26 frames com previsão e entregou MP4 com suporte a pedidos Range.
Passaram 45 testes do PosturaAI e nove testes da integração web (54 no total).
