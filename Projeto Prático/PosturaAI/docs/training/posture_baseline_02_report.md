# Segundo treino do classificador de postura — 2026-10-06

Treino concluído em `outputs/classification/posture_baseline_02/`.
O modelo ativo na app continua a ser **posture_baseline_01**. A versão 2 foi
guardada como candidata experimental e não foi promovida.

## Dados e protocolo

Oito gravações únicas, seis grupos de validação e **200 janelas de um segundo**:
136 de boa postura e 64 de má postura. Originais e clips derivados não foram
contados duas vezes. Introduções, transições, gaps e janelas que atravessam
rótulos foram excluídos. A origem dos três tutoriais de pista foi agrupada
conservadoramente; não foi inferida identidade de atletas.

As extrações anteriores foram reutilizadas após verificar hash do vídeo,
checkpoint e parâmetros. Os três vídeos mais recentes foram processados na
GPU. O RTMPose continua congelado na etapa 1; apenas o classificador geométrico
foi retreinado, com os mesmos 35 descritores, regularização e limiares.

Validação leave-one-group-out, com imputação/normalização ajustadas apenas aos
dados de treino de cada fold. O modelo final usa todas as fontes anotadas.
O vídeo de ritmos permaneceu fora dos rótulos, treino e normalização.
A cópia de `annotations.json` guardada na pasta da execução fixa os intervalos
usados neste treino.

## Resultados

| Indicador | Treino 1 | Treino 2 |
| --- | ---: | ---: |
| Gravações / grupos | 4 / 4 | 8 / 6 |
| Janelas | 102 | 200 |
| Acurácia | 51,96% | 52,00% |
| Acurácia equilibrada | 39,12% | 51,47% |
| Macro F1 | 0,390 | 0,500 |
| Recall de má postura | 21,43% | 50,00% |
| Precisão de má postura | 7,32% | 33,33% |

A composição da avaliação mudou; estes totais não são uma comparação
controlada de generalização. Nas 102 janelas das fontes originais, as previsões
fora de grupo do treino 2 tiveram acurácia **45,10%** e acurácia equilibrada
**29,14%**, abaixo dos 51,96% e 39,12% anteriores.

O grupo do tutorial de apoio do pé recebeu apenas previsões de má postura,
incluindo os exemplos assinalados como bons. A avaliação global continua
próxima dos 50% de acurácia equilibrada esperados de uma regra constante.
A melhoria global não demonstra classificação fiável; a regressão nas fontes
originais justifica conservar o modelo ativo.

Os rótulos refletem demonstrações dos autores e a classificação do utilizador;
não são uma avaliação clínica. Medidas 2D dependem do plano da câmara. Janelas
sobrepostas são correlacionadas e oito vídeos ainda são uma amostra pequena.

## Teste visual

Inferência e conversão H.264 concluídas e verificadas nos **248 frames** do vídeo
de ritmos: 171 frames com previsão, 77 sem previsão e 60 inconclusivos.
As previsões apresentadas são 100 frames de má postura e 11 de boa postura.
Não existe ground truth deste vídeo; estes números não medem acertos.

A versão candidata pode ser consultada na app em
`/postura-ai?job=f2402760f1204912937ead13ec3c1dc0`, sem alterar o modelo dos novos uploads.

## Reprodução

Na raiz PosturaAI:

```bash
.venv/bin/python -m scripts.prepare_classifier_data --annotations outputs/classification/posture_baseline_02/annotations.json --test-video "data/videos/test/What Different Running Paces Look Like #shorts.mp4" --run-dir outputs/classification/posture_baseline_02
.venv/bin/python -m scripts.train_classifier --run-dir outputs/classification/posture_baseline_02 --annotations outputs/classification/posture_baseline_02/annotations.json
.venv/bin/python -m scripts.classify_video --model outputs/classification/posture_baseline_02/classifier.json --poses outputs/classification/posture_baseline_02/test/4116077361ff1b20/pose.poses.jsonl --pose-video outputs/classification/posture_baseline_02/test/4116077361ff1b20/pose.mp4 --output outputs/classification/posture_baseline_02/test/4116077361ff1b20/classified.mp4
```

Artefactos: `classifier.json`, `annotations.json`, `training_report.json/md`,
`labelled_windows.jsonl`, `comparison.json`, `promotion_decision.json` e a pasta
`test/`. Resultados e vídeos não são versionados no Git.
