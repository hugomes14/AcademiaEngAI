# Primeiro classificador de postura

Execução: `outputs/classification/posture_baseline_01/`.
RTMPose-S mantém os pesos da etapa 1 e extrai os pontos. O novo classificador
é uma regressão logística com regularização L2, usando 35 resumos geométricos
de janelas de um segundo: joelho, anca, tornozelo, cotovelo, ombro, inclinação
absoluta do tronco e ângulo cabeça–tronco. Os lados são agrupados para reduzir
dependência de espelhamento. Não recebe texto, imagem, IDs ou velocidade de
reprodução. Os ângulos continuam a ser projeções 2D dependentes da câmara.

## Rótulos corrigidos

Fonte: [annotations.json](../../data/running_posture/annotations.json).
Por indicação do utilizador, corrigiram-se os tutoriais originalmente colocados
em `ma_postura/`, que mostram exemplos de ambas as classes. Foram inspecionadas
marcas vermelhas/verdes a intervalos de meio segundo, usando limites conservadores.

* Jim Walmsley e Paul Chelimo: rótulo bom fornecido pelo utilizador.
* Matthew Alty: cruz vermelha de 1,4–4,1 s; visto verde de 8,0–11,55 s.
  Demonstrações paradas e transições excluídas.
* `videoplayback.mp4`: trechos separados conforme cruz/visto de posição da
  cabeça, ombros, joelho e pé; restantes timestamps excluídos.

Esses rótulos refletem exemplos dos autores/utilizador. Não são confirmação
clínica nem ground truth 3D. Janelas que atravessam rótulos, gaps, mudanças de
track ou falta de pontos são excluídas. Seleciona-se a pessoa de maior área
de bbox em cada frame, uma heurística que ainda pode errar em multidões.

## Treino e validação

102 janelas: 88 boas, 14 más, quatro grupos de gravação/atleta.
Distribuição: Jim 42 boas; Paul 34 boas; Matthew 11 boas/8 más;
`videoplayback` 1 boa/6 más. Janelas têm passo 0,25 s e sobrepõem-se;
os 102 exemplos não são 102 observações independentes.

Validação leave-one-group-out, sem misturar clips/janelas da mesma fonte entre
treino e validação. Imputação e normalização são ajustadas apenas ao treino
de cada fold. Peso equilibrado por classe e grupo; regularização fixa 0,1,
sem seleção de parâmetros a partir do teste visual. O modelo final usa todas
as fontes rotuladas depois da avaliação. Identidades de origem foram inferidas
dos nomes dos ficheiros quando disponíveis; não existe reidentificação global.

Resultados fora de grupo: accuracy **51,96%**, balanced accuracy **39,12%**,
macro F1 **0,390**. O recall da classe má é **21,43%**. O desempenho é fraco e
não demonstra uma distinção fiável de boa/má postura. O relatório JSON também
contém precision, recall, F1 e average precision (área PR em degraus) por classe.

O vídeo de ritmos ficou fora dos rótulos, da normalização e do treino. O teste
gera previsões visuais, sem calcular accuracy porque não tem rótulos reais.

## Comandos reproduzíveis

Na raiz PosturaAI, extrair ou reutilizar poses verificadas por hash:

```bash
.venv/bin/python -m scripts.prepare_classifier_data \
  --run-dir outputs/classification/posture_baseline_01 \
  --test-video "data/videos/test/What Different Running Paces Look Like #shorts.mp4"

.venv/bin/python -m scripts.train_classifier \
  --run-dir outputs/classification/posture_baseline_01

.venv/bin/python -m scripts.classify_video \
  --model outputs/classification/posture_baseline_01/classifier.json \
  --poses outputs/classification/posture_baseline_01/test/4116077361ff1b20/pose.poses.jsonl \
  --pose-video outputs/classification/posture_baseline_01/test/4116077361ff1b20/pose.mp4 \
  --output outputs/classification/posture_baseline_01/test/4116077361ff1b20/classified.mp4
```

`prepare_classifier_data` só reutiliza extrações compatíveis com os hashes e
parâmetros; para um novo conjunto/modelo, usar outro `--run-dir`. As extrações
de treino conservam poses/features sem guardar vídeos redundantes. O treino
não instala dependências nem inicia o treino de RTMPose.

## Artefactos e limites

* `classifier.json`: pesos, normalizador, contrato de features e proveniência.
* `labelled_windows.jsonl`: intervalos, descritores e previsões fora de grupo.
* `training_report.json` / `.md`: distribuições, folds e métricas.
* `video_catalog.json`: hashes/origens e papel de cada vídeo.
* `dataset/<video_id>/`: pontos, features e metadados de extração.
* `test/<video_id>/classified.mp4`: vídeo com pontos e classe experimental.
* `classified.predictions.jsonl` / `.json`: scores/janelas e resumo de cobertura.

As previsões só começam quando há um segundo contíguo da pessoa dominante.
O score é disponibilizado a partir do fim da janela, sem olhar para frames
futuros, e expira após meio segundo sem nova janela ou mudança de pessoa/segmento.
Scores entre 0,4 e 0,6 são apresentados como inconclusivos. São scores sigmoid
sem calibração de confiança. As pausas de deteção aparecem sem previsão.

Para melhorar, recolher sobretudo mais exemplos negativos e positivos dos
mesmos atletas, com origens independentes e planos variados, e rever critérios
de rotulagem por problema específico. Quatro vídeos curtos, um deles em câmara
lenta e com overlays, têm fortes diferenças de origem e não cobrem o problema.
