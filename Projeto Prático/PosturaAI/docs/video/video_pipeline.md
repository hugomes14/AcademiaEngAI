# Vídeo, medidas e preparação de sequências

O pipeline aceita qualquer plano da câmara. `--view` é opcional e pode ser
`side`, `front`, `rear`, `oblique`, `mixed` ou `unknown` (predefinição).
Não existe ainda reconhecimento automático do plano. Um vídeo com vários
planos deve conservar `mixed` ou `unknown`; não deve receber uma interpretação
lateral global só porque alguns fotogramas são laterais.

```bash
cd "/home/hugomes/projects/AI Eng/Projeto Prático/PosturaAI"
.venv/bin/python -m scripts.infer_video \
  --checkpoint outputs/stage1_experimental/best.pth \
  --input "data/videos/test/A man running - side view - Animation reference - CG Reference (1080p).mp4" \
  --output outputs/videos/side_runner/side_runner_tracked_nms.mp4
```

## Tracking e suavização

Deteção SSDLite320/MobileNetV3 → RTMPose-S → associação Hungarian/IoU → EMA por
pessoa e ponto → vídeo e keypoints da mesma execução. A associação tem limiar
IoU 0,3 e conserva IDs até 5 frames de ausência. IDs são locais ao vídeo, não
identidades de atletas. Pessoas cruzadas ou saltos de posição podem trocar IDs;
o método é uma primeira implementação sem reidentificação por aparência.
Antes da associação, NMS adicional por pessoa (IoU 0,3) suprime duplicados
observados neste vídeo. Em cenas com atletas sobrepostos, pode suprimir uma
pessoa real; ajustar `--person-nms` ou usar 1 para o desativar.

EMA: `suavizado = 0,65 * atual + 0,35 * anterior`. `--ema-alpha 1` desativa a
suavização. A EMA introduz atraso e pode atenuar picos reais; conservar sempre
os pontos originais. A máscara de validade exige score ≥ 0,3, coordenadas
finitas e dentro da imagem. Confiança não equivale a precisão calibrada.
Os scores SimCC preservados podem ser superiores a 1; não são probabilidades.
Pontos ausentes não são interpolados nem desenhados. Depois de uma ausência
de um ponto, a suavização desse ponto recomeça. Depois de um frame sem deteção,
reiniciam-se suavização e segmento temporal, mesmo quando o ID é recuperado.

Cortes são estimados pela diferença média absoluta entre thumbnails RGB
64×36, limiar 0,35. Reiniciam todos os tracks sem reutilizar IDs. É uma
heurística: flashes podem produzir falsos cortes, e cortes visualmente
semelhantes podem escapar. `--cut-threshold 1` desativa a heurística.

## Artefactos

Para `side_runner_tracked_nms.mp4`:

* `.mp4`: vídeo anotado, sem áudio, mesma resolução e FPS nominal.
* `.json`: proveniência, hashes, parâmetros, cobertura de deteção, pontos em
  falta, quantidade de tracks/cortes e desempenho.
* `.poses.jsonl`: metadados na primeira linha; uma linha por frame, incluindo
  frames sem pessoas. Cada pessoa contém `track_id`, `segment_id`, bbox,
  score de deteção, `keypoints_raw` e `keypoints` suavizados (30×3), e `valid`.
  Coordenadas indisponíveis são `null`; o score original é preservado.
* `.features.jsonl`: coordenadas normalizadas e medidas por pessoa/frame,
  motivos dos valores indisponíveis e contexto de interpretação.
* `.features.csv`: uma linha por pessoa/frame, útil para exploração.
* `.features.summary.json`: distribuições das medidas e deslocamento entre
  frames antes/depois da EMA. Menor deslocamento não prova melhor precisão.
* `.features.png`: gráficos de joelhos, inclinação do tronco na imagem e
  cobertura de pontos, separados por track/segmento.

O relógio usa `frame / FPS nominal`: entradas com FPS variável devem ser
convertidas previamente para FPS constante para medidas temporais fiáveis.
A deteção de bbox junto à borda é apenas um indicador de possível corte do
corpo; não confirma que todos os membros estejam visíveis.

## Medidas implementadas

As fórmulas versionadas ficam no cabeçalho das features (`feature_spec`).

* Ângulos interiores projetados de joelho, anca, tornozelo, cotovelo e ombro,
  dos dois lados, exportados em todos os planos, quando os pontos são válidos.
  Ângulo do joelho de 180° significa os segmentos alinhados na imagem.
  Ângulos de anca/tornozelo são proxies entre segmentos, não flexão clínica
  nem dorsiflexão calibrada. O tornozelo usa o primeiro metatarsal do SRKD.
* Inclinação do tronco na imagem, de Pelvis a UpTrunk, disponível em qualquer
  plano; sinal positivo para a direita do ecrã.
* Inclinação para a direção de corrida e distância horizontal tornozelo–anca:
  exigem vista lateral e direção conhecida; senão ficam `null`.
* Inclinação lateral do tronco e da linha das ancas: vista frontal/traseira
  conhecida; senão `null`. A linha das ancas não é automaticamente hip drop.
* Velocidades dos ângulos projetados (graus/s) e dos tornozelos relativamente
  à pelvis (comprimentos de tronco/s), apenas entre frames válidos contíguos.
* Posição vertical da pelvis em píxeis apenas quando `--camera-motion fixed`
  foi explicitamente declarado. Não é oscilação vertical do centro de massa;
  movimento da câmara, zoom e perspetiva interferem.

Normalização: subtrair Pelvis e dividir pela distância Pelvis–UpTrunk;
coordenadas x para a direita e y para baixo. Se pelvis ou tronco estiverem
indisponíveis/degenerados, toda a normalização fica indisponível. Os ângulos
continuam disponíveis quando os seus próprios pontos são válidos.

`visible_side` é opcional. Os dois lados são exportados, mas o membro oculto
tem risco de erro mesmo com confiança alta. Não há métricas em metros nem
reconstrução 3D. A interpretação sagital exige vista lateral; a geometria
projetada fica disponível em planos desconhecidos sem essa interpretação.
A validade de medidas 2D versus 3D varia por medida e participante, como
documentado neste [estudo de corrida](https://pubmed.ncbi.nlm.nih.gov/32705424/).

Recalcular features sem GPU, deteção ou RTMPose:

```bash
.venv/bin/python -m scripts.extract_features \
  --input outputs/videos/side_runner/side_runner_tracked_nms.poses.jsonl
```

Se mais tarde houver contexto conhecido, usar `--view side
--running-direction left` e um `--output` diferente para preservar a versão
original. Não declarar câmara fixa numa gravação que acompanha o corredor.

## Dataset de atletas e futuros rótulos

Recolher vídeos com diversidade lateral (ambos os lados), frontal, traseira,
oblíqua e planos mistos, além de atletas, velocidades, roupa e iluminação
variados. Não restringir a coleta pela vista. Quando conhecida, anotá-la como
metadado para analisar cobertura e desempenho por plano; desconhecido é válido.

Identificar atletas com `--athlete-id` e a gravação original com `--source-id`
quando disponíveis. Clips da mesma gravação partilham source_id. O hash do
ficheiro identifica o vídeo, mas não relaciona automaticamente clips editados.
Não usar `track_id` como identidade global nem assumir que profissional = boa
postura. O ficheiro [video_manifest.csv](../../data/running_posture/video_manifest.csv)
é um modelo de catálogo; os campos podem ser preenchidos progressivamente.

Gerar janelas contíguas de 2 segundos, passo de 1 segundo:

```bash
.venv/bin/python -m scripts.build_sequences \
  --input outputs/videos/side_runner/side_runner_tracked_nms.features.jsonl \
  --output outputs/videos/side_runner/side_runner_sequences.jsonl
```

As janelas contêm `T×30×2` pontos normalizados, máscaras, medidas, identidade
de origem e `labels: null`, `split: null`. A confiança e coordenadas originais
continuam no JSONL de poses. Não se atravessam gaps nem tracks/segmentos.
Descartar por predefinição janelas com menos de 60% dos pontos válidos.

Isto prepara dados, mas não treina o classificador. Rótulos revistos de
critérios específicos e exemplos variados de cada classe continuam necessários.
O primeiro treino experimental usa um percurso separado, com intervalos
revistos em `data/running_posture/annotations.json`; consultar
[classificador](classifier.md) para comandos e resultados.
Separar treino/validação/teste por atleta e gravação original, incluindo todas
as janelas sobrepostas e vários planos do mesmo atleta no mesmo split; nunca
sortear fotogramas isolados. Sem identidade conhecida, agrupar por origem e
registar a limitação. Avaliar desempenho por plano conhecido e desconhecido.
Sinais de contacto no chão, cadência validada, overstriding e hip drop por fase
ainda não estão implementados; não gerar diagnósticos com estes proxies.
