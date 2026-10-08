# Vídeos para o dataset de postura de corrida

Colocar os vídeos nas pastas:

- `boa_postura/`: exemplos que classificaste como boa postura.
- `ma_postura/`: exemplos que classificaste como má postura.

Aceitar vídeos de vários planos, incluindo planos mistos ou desconhecidos.
Quando possível, usar nomes que identifiquem atleta e gravação, por exemplo
`atleta_001_gravacao_01_clip_01.mp4`. Clips e planos do mesmo atleta/gravação
ficam no mesmo grupo quando se preparar treino, validação e teste.

As pastas registam a tua classificação inicial do vídeo. Se a postura mudar
no decorrer de uma gravação, dividir em clips com rótulo consistente.
O pipeline de extração de pontos e medidas está em `docs/video/video_pipeline.md`.
Os intervalos efetivamente usados no treino estão em `annotations.json`.
Vídeos de tutoriais com exemplos bons e maus conservam o ficheiro original,
mas têm rótulos diferentes por trecho. Os trechos de transição ficam excluídos.
Para novos vídeos, preencher o ficheiro de anotações com hash, origem e intervalos.

Os comandos de extração, treino e teste do primeiro classificador estão em
[classificador](../../docs/video/classifier.md). As pastas expressam a classificação
inicial; `annotations.json` é a fonte de rótulos revista para o treino.

## Importação de 2026-10-06

Os dois ficheiros acrescentados a `boa_postura/` são cópias exatas de fontes já
existentes, verificadas por SHA-256 e registadas no catálogo:

- `videoplayback.mp4`: igual ao ficheiro em `ma_postura/`. Já tem intervalos de
  boa e má postura em `annotations.json`; contar uma única vez no treino.
- `What Different Running Paces Look Like #shorts.mp4`: igual ao vídeo reservado
  em `data/videos/test/`. Conserva o papel de teste visual, sem rótulos de postura
  definidos. As legendas indicam ritmos e não certificam a postura. Não incluir
  no treino por estar nesta pasta. Se passar a fonte de treino, definir os
  intervalos e escolher outro vídeo independente para teste.

Essa primeira importação manteve quatro gravações únicas e não alterou os
rótulos existentes.

### Ficheiros corrigidos posteriormente

Foi acrescentado `Knee drive while running! #runningtips ❌✅.mp4`, uma gravação
nova com exemplos negativos e positivos assinalados pelo autor. A revisão dos
frames a cada 0,5 s originou intervalos conservadores em `annotations.json`:

- **1,0–4,7 s:** `ma_postura`, cruz vermelha e texto “No knee drive”.
- **5,5–10,5 s:** `boa_postura`, visto verde e texto “Knee drive”.

Os restantes tempos ficam excluídos: introdução, transições e ausência de
marcações. Os rótulos descrevem o exemplo de elevação do joelho apresentado
pelo autor; não certificam a técnica global da pessoa.

`videoplayback (1).mp4` tem o mesmo SHA-256 do `videoplayback.mp4` já anotado;
as três cópias nas pastas boa/má representam uma única gravação. A cópia do
vídeo de ritmos foi removida de `boa_postura/`; o original permanece em teste.

O conjunto anotado passa a **cinco gravações únicas**. O classificador atualmente
usado na app continua a ser o primeiro treino, com quatro gravações; estas
anotações só serão incorporadas no modelo após novo treino.

### Mais três vídeos separados

Foram revistos mais três vídeos com marcações explícitas dos autores. Os
originais foram preservados e foram gerados seis clips H.264 sem áudio em
`clips/boa_postura/` e `clips/ma_postura/`:

| Vídeo | Má postura (s) | Boa postura (s) |
| --- | --- | --- |
| Common running mistake! Over-Striding! | 1,0–6,0 | 7,5–11,5 |
| Proper foot strike | 0,0–5,0 | 5,5–10,6 |
| Running Form Pro Tip | 2,0–4,5 | 8,5–10,5 |

Os limites são conservadores, após observação dos frames a cada 0,5 s. As
transições e as marcações incompletas ficam excluídas. Os rótulos referem os
exemplos dos autores sobre passada/apoio do pé, não uma avaliação global ou
clínica da postura.

`annotations.json` conserva os intervalos, a evidência e o caminho/hash de cada
clip. `clips/manifest.json` e o [registo da separação](../../docs/data/clip_separation_2026-10-06.json)
conservam a proveniência. Os clips são derivados para consulta; o treino usa
os originais com os intervalos de `annotations.json`, evitando contar cada
sequência duas vezes. Não juntar clips do mesmo original em splits diferentes.

Os dois novos tutoriais filmados na pista e o tutorial Knee drive ficam no
mesmo grupo de validação, de forma conservadora, pelo cenário e formato
semelhantes. A origem comum está por confirmar; não foi inferida a identidade
de atletas. O vídeo Proper foot strike fica num grupo próprio.

O conjunto passa a **oito gravações únicas anotadas**. A adição e separação
destes vídeos não retreina nem substitui automaticamente o modelo da app.

O [segundo treino](../../docs/training/posture_baseline_02_report.md) foi depois
executado com estas oito gravações: 200 janelas válidas, em seis grupos.
A versão candidata foi guardada; a app mantém o primeiro classificador devido
à validação ainda fraca e à regressão nas fontes originais.

Posteriormente, a pedido do utilizador, a app passou para **Extra Trees**,
treinado nas mesmas 200 janelas. A [comparação de modelos](../../docs/training/nonlinear_classifier_report.md)
conserva a validação e as limitações; a descrição anterior refere a decisão
tomada no fim do segundo treino de regressão logística.
