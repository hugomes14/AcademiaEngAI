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
