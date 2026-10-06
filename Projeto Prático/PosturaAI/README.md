# PosturaAI

Serviço de análise de corrida em vídeo da plataforma RunningAI. Usa RTMPose com os 30 keypoints do Synthetic Runner Keypoint Dataset (SRKD), tracking, medidas 2D e um classificador experimental de postura.

A integração na app principal está disponível em `/postura-ai`: importar um vídeo, acompanhar o progresso e os frames de pose e reproduzir ou descarregar o resultado anotado. Consulta as [instruções da app](../INSTRUCOES_APP.md). O worker mantém as dependências de GPU no ambiente dedicado deste projeto.

Está guardada a [proposta de análise comentada com LLM local](docs/plans/local_llm_feedback.md): LM Studio para explicar as previsões com base nas medidas e contribuições do classificador; sugestões sustentadas em conhecimento revisto; Unsloth para eventual ajuste posterior. Esta integração ainda não foi implementada.

O [plano de implementação](docs/plans/rtmpose_srk_implementation_plan.md) define arquitetura, critérios de avanço e artefactos esperados. O desenvolvimento avançou de M0 para uma experiência M1 sem auditoria, por decisão do utilizador. A etapa 1 terminou, foi avaliada no teste sintético e é a referência atual para o pipeline de vídeo. As etapas 2/3 não foram executadas.

## Estado em 2026-10-06

Validação completa do SRKD aprovada, com as 18 boxes e 84 pontos fora da imagem já conhecidos. Estão implementados os comandos M0, as métricas NME/PCK, o treino RTMPose-S por etapas, inferência em imagem e vídeo, tracking, suavização e primeiras medidas 2D.

O smoke concluiu em `outputs/archive/training/smoke_experimental_04/`. A etapa 1 terminou 10 épocas em `outputs/stage1_experimental/`; a época 10 teve NME de validação 0,01476, PCK 97,73%, NME dos pés 0,02161 e PCK dos pés 95,26%. O teste em 9 298 imagens apresentou NME 0,01242, PCK 98,87%, NME dos pés 0,01647 e PCK dos pés 98,20% (`outputs/stage1_test/test_metrics.json`). São resultados sintéticos de splits sem auditoria, não métricas de vídeo real. O [relatório do primeiro treino](docs/training/first_training_report.md) conserva o registo histórico do arranque.

O manifesto contém 153 grupos, todos pendentes de revisão. A distribuição **provisória** com seed 42 contém 74 278 imagens de treino, 9 248 de validação e 9 298 de teste. A auditoria gerou 205 pares para revisão (101 limites, 100 pares entre splits e 4 candidatos visuais adicionais), incluindo 9 casos ambíguos. Existem 100 overlays de anotações e 21 folhas de comparação.

Por decisão do utilizador, o desenvolvimento prossegue **sem auditoria manual**, através do modo experimental descrito abaixo. As revisões permanecem pendentes, mas deixam de bloquear esta experiência. Os splits experimentais usam a mesma atribuição de grupos e mantêm as correções/validações geométricas. A `.venv` dedicada usa Python 3.11 e PyTorch 2.7.1/CUDA 12.8; treino e inferência RTMPose foram executados na RTX 5080 fora do sandbox. A instalação OpenMMLab está descrita em [ambiente de treino](docs/training/training_environment.md). Consultar [preparação da revisão e hashes](docs/data/m0_preparation.md), [auditoria visual](docs/data/visual_audit.md) e [auditoria dos grupos](docs/data/split_audit.md).

## Organização

```text
PosturaAI/
├── README.md
├── requirements*.txt
├── web_service.py                  # fila local e integração com a app principal
├── configs/                         # configuração dos datasets e modelos
├── src/posturaai/                    # lógica da aplicação e dos modelos
├── scripts/                         # comandos executáveis
├── tests/                           # testes automatizados
├── data/
│   ├── srkd/                        # manifesto e splits sintéticos
│   ├── videos/test/                 # vídeos de referência
│   └── running_posture/
│       ├── boa_postura/             # vídeos classificados como boa postura
│       ├── ma_postura/              # vídeos classificados como má postura
│       └── video_manifest.csv       # catálogo dos vídeos
├── docs/
│   ├── README.md                    # índice da documentação
│   ├── plans/                       # plano de implementação
│   ├── data/                        # especificação e preparação do SRKD
│   ├── training/                    # ambiente e relatórios de treino
│   └── video/                       # pipeline e relatórios de vídeo
├── outputs/
│   ├── pretrained/                 # pesos oficiais
│   ├── stage1_experimental/         # treino atual e best.pth
│   ├── stage1_test/                 # avaliação sintética
│   ├── smoke_experimental_04/       # smoke concluído
│   ├── videos/                      # resultados agrupados por vídeo
│   ├── classification/              # datasets, modelos e testes de postura
│   ├── web/jobs/                    # uploads e resultados da app, por análise
│   ├── visualizations/              # imagens, anotações e galeria SRKD
│   └── archive/                     # tentativas e resultados antigos
├── Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/
└── Synthetic-Runner-Keypoint-Dataset-2026-06-07/
```

As duas árvores originais do SRKD correspondem aos caminhos registados nos
splits e na proveniência do treino. Vídeos novos de boa/má postura entram em
`data/running_posture/`; os resultados de cada vídeo ficam numa pasta própria
em `outputs/videos/`. O [índice da documentação](docs/README.md) reúne os guias.

## Avançar sem auditoria

```bash
source .venv/bin/activate
python -m scripts.prepare_splits --seed 42 --unaudited
python -m scripts.train --stage smoke --run-id smoke_experimental
python -m scripts.train --stage 1 --allow-unaudited --run-id stage1_experimental
python -m scripts.train --stage 2 --allow-unaudited --run-id stage2_experimental --init-checkpoint outputs/stage1_experimental/best.pth
python -m scripts.train --stage 3 --allow-unaudited --run-id stage3_experimental --init-checkpoint outputs/stage2_experimental/best.pth
```

O primeiro comando produz `train.json`, `val.json`, `test.json`, `corrections.csv` e `preparation_report.json` em `data/srkd/derived/`. O relatório regista `audit_status=not_reviewed`, `experimental=true`, ausência de revisor e hashes dos inputs/outputs. As 18 boxes são limitadas à imagem e os 84 pontos fora são ocultados apenas nas cópias derivadas. O manifesto e as decisões mantêm-se pendentes.

O treino experimental exige `--allow-unaudited` nas etapas completas e verifica se os ficheiros correspondem ao relatório de preparação. O smoke guarda também a proveniência experimental. Os metadados de cada treino conservam esse estado. Métricas desta experiência devem ser apresentadas como resultados de splits sem auditoria, com a possível sobreposição de cenários já documentada. A sequência acima requer o runtime GPU e o sucesso da etapa precedente; não instala dependências.

Os diretórios das execuções existentes estão preservados; ao repetir um treino, escolher outro `--run-id` ou retomar com `--resume-run`. Os pesos do smoke são apenas de diagnóstico e não inicializam a etapa 1.

## Inferência numa imagem

```bash
.venv/bin/python -m scripts.infer_image --checkpoint outputs/stage1_experimental/best.pth --input Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/Images/000410.jpeg
```

Nas imagens originais SRKD, o comando encontra e limita a box de referência. Para outra imagem, fornecer `--bbox X Y WIDTH HEIGHT`. O comando produz um JPEG anotado e um JSON com 30 triplos `(x, y, score)`, nomes dos pontos e proveniência do checkpoint. `--device cpu` permite executar sem GPU. A deteção automática está disponível no comando de vídeo abaixo.

## Vídeo, tracking e medidas

```bash
.venv/bin/python -m scripts.infer_video \
  --checkpoint outputs/stage1_experimental/best.pth \
  --input "data/videos/test/A man running - side view - Animation reference - CG Reference (1080p).mp4" \
  --output outputs/videos/side_runner/side_runner_tracked_nms.mp4
```

Produz vídeo sem áudio, pontos originais/suavizados por frame, IDs e segmentos, medidas 2D, CSV e gráficos. Aceita qualquer plano sem exigir anotação da vista; `--view` é opcional. Não classifica postura. Os comandos para recalcular features e criar sequências sem repetir a inferência estão em [pipeline de vídeo](docs/video/video_pipeline.md), juntamente com fórmulas, limitações e organização do dataset com planos variados.

## Classificador experimental

Foi treinado um primeiro classificador geométrico com 102 janelas de quatro
vídeos, após corrigir os rótulos por trecho nos tutoriais mistos. A validação
por gravação apresentou balanced accuracy 39,12% e macro F1 0,390, resultados
fracos que ainda não sustentam avaliação fiável de postura. O vídeo de ritmos
foi reservado para inferência visual, sem rótulos de teste. Pesos, dados e
relatórios estão em `outputs/classification/posture_baseline_01/`; consultar
[classificador e comandos](docs/video/classifier.md).

## Dados locais

O JSON bruto está em `Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/keypoints_srkd.json`. As 92 824 imagens estão distribuídas pelas duas árvores `Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/` e `Synthetic-Runner-Keypoint-Dataset-2026-06-07/`; não são copiadas para `data/`. Os ficheiros brutos e derivados grandes não são versionados. O manifesto de grupos auditado pode ser versionado.

## Preparação

Usar um ambiente próprio do PosturaAI. O ambiente da aplicação Flask não contém as dependências de pose.

```bash
cd 'Projeto Prático/PosturaAI'
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.validate_dataset
.venv/bin/python -m scripts.visualize_annotations --samples 100 --output outputs/visualizations/ground_truth
.venv/bin/python -m unittest discover -s tests -v
```

Para o percurso auditado, rever as visualizações e seguir a sequência de construção e auditoria dos grupos no plano. Para a experiência autorizada sem auditoria, usar `--unaudited`. O treino RTMPose depende de uma GPU local e de um conjunto compatível PyTorch/CUDA/OpenMMLab, ainda por fixar.

## Percurso alternativo: retomar a auditoria M0

Os artefactos provisórios já foram gerados localmente. Para os reproduzir, depois de construir o manifesto com `python -m scripts.build_groups`:

```bash
python -m scripts.prepare_splits --seed 42 --provisional
python -m scripts.audit_groups --output docs/data/split_audit.md
python -m scripts.build_audit_gallery
```

Abrir `outputs/visualizations/m0_review.html` no navegador. A galeria reúne as 100 anotações e os 205 pares com ligações às imagens originais; não requer servidor. Preencher `decision`, `reviewer` e `note` em `data/srkd/audit_decisions.csv`. A revisão das anotações, incluindo resize/flip, fica registada em `docs/data/visual_audit.md` e `docs/data/dataset_spec.md`.

Se houver decisões `same`, executar `python -m scripts.audit_groups --apply-decisions`, repetir a distribuição provisória e regenerar a auditoria/galeria. Só após concluir todas as revisões:

```bash
python -m scripts.audit_groups --approve --reviewer 'NOME_DO_REVISOR'
python -m scripts.prepare_splits --seed 42 --final
```

A aprovação dos grupos não substitui a revisão anatómica das anotações: ambas são necessárias para fechar M0. Os CSVs de decisões e distribuição provisória, caches e JSONs derivados continuam locais; guardar a evidência e os hashes ao fechar a auditoria. A configuração SRKD e o manifesto têm exceções explícitas no `.gitignore` para permitir versionamento.
