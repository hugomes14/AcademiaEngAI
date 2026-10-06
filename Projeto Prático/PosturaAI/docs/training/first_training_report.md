# Primeiro treino experimental — 2026-10-06

**Estado deste registo:** smoke concluído; etapa 1 em curso, com a primeira época validada. As métricas abaixo são preliminares e usam splits sem auditoria, conforme a decisão do utilizador. Não representam uma avaliação final nem resultados em vídeo real.

## Dados e ambiente

Foram exportados os três splits (74 278 / 9 248 / 9 298 imagens), preservando os 153 grupos e aplicando 102 correções derivadas: 18 boxes e 84 pontos fora da imagem. O JSON original e o manifesto mantêm os hashes anteriores. As decisões de auditoria permanecem vazias; não foi criada uma aprovação fictícia.

O ambiente dedicado usa Python 3.11.17, PyTorch 2.7.1+cu128, MMCV 2.1.0 com operadores CPU, MMEngine 0.10.7, MMDetection 3.3.0 e MMPose 1.3.2. RTMPose executa na NVIDIA GeForce RTX 5080 Laptop GPU. As versões e o hardware estão em [runtime_preflight.json](runtime_preflight.json); os pacotes resolvidos estão em [requirements-lock.txt](../../requirements-lock.txt).

O arranque exigiu corrigir a leitura eager da metainfo SRKD, configurar CuBLAS para determinismo e permitir explicitamente tipos NumPy/MMEngine na leitura restrita de checkpoints PyTorch. Foram conservados os registos das tentativas iniciais que falharam antes das épocas.

## Smoke

Execução: `outputs/archive/training/smoke_experimental_04/`. Três épocas, 64 imagens fixas, batch físico 16 e acumulação 4. A verificação de forward produziu saídas finitas de shapes `(1,30,384)` e `(1,30,512)`. A avaliação do smoke usa as mesmas 64 imagens de treino, apenas para diagnóstico.

| Época | Loss | NME | NME dos pés |
| ---: | ---: | ---: | ---: |
| 1 | 0,293328 | 0,534042 | 0,540606 |
| 2 | 0,291909 | 0,517289 | 0,628453 |
| 3 | 0,290130 | 0,513889 | 0,612299 |

O smoke confirmou integração, treino e gravação/recarregamento do checkpoint. Três atualizações do otimizador não bastam para demonstrar bom overfit ou generalização; a perda diminuiu ligeiramente e o erro dos pés não melhorou. Os seus pesos não foram usados para inicializar a etapa 1.

## Etapa 1 — primeira época

Execução em curso: `outputs/stage1_experimental/`. Backbone e estatísticas BN congelados, cabeça com 30 canais, LR inicial `1e-4`, AdamW, batch físico 16, batch efetivo 64, quatro workers e seed 42. Inicialização a partir do backbone oficial, com hash registado nos metadados. Máximo de 10 épocas e paragem após quatro validações sem melhoria.

| Métrica de validação | Época 1 |
| --- | ---: |
| Imagens | 9 248 |
| NME global | 0,0188 |
| PCK@0,05 global | 96,40% |
| NME dos pés | 0,0288 |
| PCK@0,05 dos pés | 92,07% |
| Pontos visíveis | 277 440 |
| Pontos visíveis dos pés | 92 480 |

NME normaliza o erro pela altura da box GT derivada; PCK usa o limiar 0,05 dessa altura. A possível sobreposição de cenários já identificada permanece sem resolver nesta experiência. AP/OKS não foram calculados; o conjunto de teste não foi avaliado.

O primeiro checkpoint foi recarregado na GPU e usado para inferência top-down na imagem SRKD 000410 com box de referência. Foram exportados 30 pontos/confianças e uma imagem anotada, inspecionada para confirmar funcionamento da renderização:

- `outputs/visualizations/inference/000410_stage1_experimental_epoch_1.jpg`
- `outputs/visualizations/inference/000410_stage1_experimental_epoch_1.json`

Este exemplo não constitui auditoria anatómica ou uma revisão do dataset. O skeleton continua identificado como não revisto.

## Artefactos e continuação

- Configuração, logs, checkpoints e estado da execução: `outputs/stage1_experimental/`.
- Metadados do smoke: `outputs/archive/training/smoke_experimental_04/run_metadata.json`.
- Proveniência dos dados: `data/srkd/derived/preparation_report.json` e [experimental_preparation.json](../data/experimental_preparation.json).
- Inferência: `python -m scripts.infer_image --help`.
- Verificação: **23 testes passaram** no ambiente Python 3.11; `git diff --check` passou.

A etapa 1 continua automaticamente. Ao terminar, a CLI guarda `best.pth` pelo menor NME de validação e `latest.pth`, além de atualizar `run_metadata.json`. Os checkpoints numerados antigos podem ser removidos pela retenção automática. As etapas 2/3 e a avaliação final continuam por executar; o comando dedicado de avaliação ainda não está implementado.
