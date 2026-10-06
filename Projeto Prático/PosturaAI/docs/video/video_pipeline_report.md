# Teste do pipeline em 2026-10-06

Entrada: `data/videos/test/A man running - side view - Animation reference - CG Reference (1080p).mp4`.
1920×1080, 25 FPS, 537 frames (21,48 segundos). Checkpoint da etapa 1:
`outputs/stage1_experimental/best.pth`, SHA256
`4db9347d5b9d6b0cb89da4dd6748a36ca313eff96eab2f4e91423990b2403732`.
Vista, direção, lado visível e movimento da câmara ficaram `unknown`, como
teste do fluxo que aceita qualquer plano sem exigir anotação prévia.

## Execução final

Artefactos: `outputs/videos/side_runner/side_runner_tracked_nms.*`, mais
`outputs/videos/side_runner/side_runner_sequences.jsonl`.

| Indicador | Resultado |
| --- | ---: |
| Frames processados e guardados | 537 |
| Frames com pessoa | 497 (92,55%) |
| Frames sem deteção | 40 |
| IDs locais | 2 |
| Segmentos contíguos | 23 |
| Cortes assinalados pela heurística | 0 |
| Observações de pessoa com features | 497 |
| Pontos excluídos pelo filtro, nos frames detetados | 0,63% |
| Janelas contíguas de 2 segundos (50 frames) | 5 |
| Testes automatizados | 36 aprovados |

A primeira versão criou 22 IDs devido a caixas duplicadas do SSDLite.
A NMS adicional por pessoa (IoU 0,3) reduziu-os a 2. O segundo ID aparece no
frame 511 depois de 11 frames sem deteção, além da retenção de 5 frames.
Não representa outro atleta e não deve ser usado como identidade global.
Os gaps reiniciam as medidas temporais; não se fabricam poses nos 40 frames.

O MP4 final reabriu com 537 frames e 25 FPS. Foram inspecionados exemplos de
sobreposição e o gráfico temporal para confirmar a renderização; isso não
mede precisão anatómica ou valida biomecânica. Foram verificadas 537 linhas
de frames nas poses/features e as cinco janelas de 50 frames sem rótulos.
O extractor independente foi executado sem GPU para confirmar reutilização
dos keypoints e gerar os gráficos atuais.

Os scores brutos médios são aproximadamente 1,08, possíveis no SimCC;
não são probabilidades nem taxa de acerto. Os dados conservam proveniência
experimental e os rótulos de postura permanecem nulos. Não há ground truth
real, medidas em metros, eventos de contacto ou classificação de postura.

As medidas são projeções 2D e podem variar com a vista. O conjunto de vídeos
deve abranger planos variados; quando houver metadados, avaliar separadamente
os planos, com split por atleta/gravação. O trabalho seguinte é melhorar a
cobertura do detetor, coletar mais vídeos e definir rótulos específicos de
postura antes do treino supervisionado.
