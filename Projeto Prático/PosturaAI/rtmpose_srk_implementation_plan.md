# PosturaAI — plano de implementação RTMPose + SRKD

**Estado:** arquitetura definida; implementação e treino pendentes.

**Raiz do projeto:** `Projeto Prático/PosturaAI/`. Todos os caminhos abaixo são relativos a esta pasta.

**Fonte de verdade:** este documento define a ordem de trabalho e os critérios de avanço. Configurações, resultados e relatórios produzidos durante a implementação devem referenciar a revisão deste ficheiro.

## 1. Entregas e limites

| Marco | Entrega | Condição para avançar |
| --- | --- | --- |
| M0 — dados | SRKD validado, especificação dos 30 pontos, anotações derivadas e splits auditados | Integridade e auditoria aprovadas |
| M1 — primeiro modelo | RTMPose-S de 30 pontos treinado, checkpoint selecionado e relatório de teste no SRKD | Critérios da secção 8 cumpridos |
| M2 — inferência real | Imagem e vídeo reais, deteção, tracking, suavização e exportação de keypoints | Qualidade espacial e temporal medida |
| M3 — biomecânica | Normalização, features e regras interpretáveis | Testes geométricos e revisão de exemplos |
| M4 — classificação temporal | Modelo supervisionado de sequências | Só iniciar com sequências e rótulos validados |
| M5 — produto | Integração do PosturaAI na aplicação Flask RunningAI | Contratos e desempenho de M2/M3 estabilizados |

**A primeira entrega termina em M1.** Usa apenas o SRKD; não exige vídeo real, imagens reais anotadas, classificador temporal nem integração Flask. M2–M5 são especificações posteriores e não bloqueiam M1. Não avançar de M0 para treino completo enquanto o split auditado não existir. O teste de overfit de M0 pode usar uma amostra fixa antes desse split, mas os seus pesos são descartados; a etapa 1 começa de novo a partir do backbone pré-treinado. Esse teste não produz métricas de generalização.

## 2. Estado conhecido do repositório e princípios

Inspeção local em 2026-09-17:

- O JSON bruto está em `Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/keypoints_srkd.json`. Contém 92 824 imagens e 92 824 anotações, uma pessoa por imagem, resolução 640 × 360, 30 keypoints por pessoa e visibilidade 2 em todos os pontos brutos.
- As imagens `000001.jpeg`–`001219.jpeg` estão em `Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/Images/`; as restantes `001220.jpeg`–`092824.jpeg` estão em `Synthetic-Runner-Keypoint-Dataset-2026-06-07/Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/Images/`. Os conjuntos não se sobrepõem e cobrem todas as referências do JSON.
- Existem 18 bounding boxes que passam até 6 píxeis abaixo da imagem. Há também 84 keypoints dos pés, em 50 anotações, marcados como visíveis embora estejam até 7 píxeis abaixo da imagem. São problemas conhecidos a tratar nas anotações derivadas, sem alterar o original.
- Não há `runner_id`, `sequence_id`, `scene_id`, `camera_id` ou equivalente no JSON. O nome numérico não comprova uma sequência; imagens próximas podem partilhar corredor/cenário. Não é possível afirmar ausência absoluta de leakage com os dados disponíveis.
- O JSON declara licença `CC BY-NC 4.0`. Registar a licença dos dados e dos pesos usados em cada treino antes de qualquer distribuição do modelo.
- Ainda não existem código, configurações, ambiente MMPose ou checkpoints do PosturaAI. A aplicação Flask da raiz apresenta PosturaAI como serviço planeado. A `.venv` da raiz usa Python 3.13 e não tem PyTorch/MMPose; o acesso à GPU não está disponível neste ambiente WSL2.

**Invariantes:** dados brutos imutáveis; splits e pré-processamento reproduzíveis; `val` serve para escolher modelos e `test` fica reservado; métricas de pés são sempre mostradas; checkpoints preservam configuração e proveniência; inferência de pose e análise biomecânica têm módulos separados. Não copiar os cerca de 7 GB de imagens para uma nova árvore.

## 3. Estrutura alvo e interfaces

```text
PosturaAI/
├── rtmpose_srk_implementation_plan.md
├── README.md
├── requirements.txt                 # dependências declaradas do serviço
├── requirements-lock.txt            # versões resolvidas no ambiente de treino
├── configs/
│   ├── datasets/srkd.py             # metainfo: 30 pontos, flip, skeleton, pesos
│   └── rtmpose/                     # base RTMPose-S e overrides por etapa
├── data/srkd/
│   ├── group_manifest.csv           # image_id, group_id, método e estado da auditoria
│   └── derived/                     # train/val/test COCO e relatório de correções
├── src/posturaai/                   # validação, preparação, métricas, treino, inferência
├── scripts/                         # CLIs finas, sem lógica de negócio duplicada
├── tests/
├── docs/                            # especificação, auditoria, treino e avaliação
├── outputs/                         # checkpoints, logs e visualizações gerados
├── Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/
└── Synthetic-Runner-Keypoint-Dataset-2026-06-07/
```

Os diretórios dos dados brutos permanecem onde estão. Os ficheiros COCO derivados usam `file_name` relativo à raiz `PosturaAI/`; a configuração MMPose usa essa raiz como `data_root` e prefixo de imagem vazio. Assim, `CocoDataset` encontra as imagens nas duas árvores sem cópia ou links. A geração deve falhar se um caminho resolver fora de `PosturaAI/` ou não corresponder a uma imagem existente. Versionar código, configs, documentação e manifesto auditado; ignorar imagens brutas, `outputs/`, ambientes, sidecars `:Zone.Identifier` e JSONs derivados grandes. Registar hashes do JSON bruto, do manifesto e dos JSONs derivados no relatório da preparação.

### Contrato dos comandos de M0/M1

Os comandos abaixo são interfaces a implementar, executadas a partir de `PosturaAI/`. Todos devem aceitar `--help`, devolver código não zero em falha e escrever um resumo legível. Os caminhos indicados são os valores por omissão.

```bash
python -m scripts.validate_dataset
python -m scripts.visualize_annotations --samples 100 --output outputs/visualizations/ground_truth
python -m scripts.build_groups --output data/srkd/group_manifest.csv
python -m scripts.prepare_splits --manifest data/srkd/group_manifest.csv --seed 42 --provisional
python -m scripts.audit_groups --manifest data/srkd/group_manifest.csv --output docs/split_audit.md
python -m scripts.prepare_splits --manifest data/srkd/group_manifest.csv --seed 42 --final
python -m scripts.train --stage smoke
python -m scripts.train --stage 1
python -m scripts.train --stage 2 --init-checkpoint outputs/stage1/best.pth
python -m scripts.train --stage 3 --init-checkpoint outputs/stage2/best.pth
python -m scripts.evaluate --checkpoint outputs/final/best.pth --split test
python -m scripts.infer_image --checkpoint outputs/final/best.pth --input IMAGE --output outputs/visualizations/inference
```

`--provisional` cria apenas a atribuição temporária de grupos aos splits para a auditoria cruzada, sem JSONs de treino. `--final` exige manifesto sem decisões pendentes e auditoria aprovada; só então produz os três JSONs COCO. `train` e `evaluate` recebem a configuração versionada correspondente à etapa. `--init-checkpoint` carrega pesos e cria novo otimizador/scheduler; uma retoma exata de uma execução interrompida usa `--resume-run` e restaura também esses estados. `infer_image` em M1 aceita uma imagem SRKD e a bounding box de referência para testar o estimador top-down; deteção automática pertence a M2.

## 4. M0 — dados, anotações e splits

### 4.1 Especificação dos keypoints

A ordem vem de `categories[0].keypoints` no JSON bruto. Índices abaixo são **0-based**; as arestas de `categories[0].skeleton` no ficheiro bruto são **1-based**. A implementação usa os nomes originais e converte as arestas uma vez ao carregar a metainfo.

| Índice | Nome | Índice | Nome | Índice | Nome |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | Nose | 10 | RightWrist | 20 | RightFirstMetatarsal |
| 1 | LEye | 11 | LeftHip | 21 | LeftFifthMetatarsal |
| 2 | REye | 12 | RightHip | 22 | RightFifthMetatarsal |
| 3 | LeftEar | 13 | LeftKnee | 23 | LeftToe |
| 4 | RightEar | 14 | RightKnee | 24 | RightToe |
| 5 | LeftShoulder | 15 | LeftAnkle | 25 | TopHead |
| 6 | RightShoulder | 16 | RightAnkle | 26 | BackHead |
| 7 | LeftElbow | 17 | LeftHeel | 27 | UpTrunk |
| 8 | RightElbow | 18 | RightHeel | 28 | MiddleTrunk |
| 9 | LeftWrist | 19 | LeftFirstMetatarsal | 29 | Pelvis |

Pares para `horizontal flip`: `(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16), (17,18), (19,20), (21,22), (23,24)`. Os índices `0,25,26,27,28,29` não têm par. A operação de flip deve ser uma involução: aplicá-la duas vezes devolve o mapeamento original.

Arestas únicas do skeleton bruto após conversão para 0-based (33; o JSON contém duas repetições):

```text
(0,1) (0,2) (1,3) (2,4) (25,26) (0,26) (0,27) (5,27)
(6,27) (27,28) (0,3) (1,2) (28,29) (11,29) (12,29)
(5,7) (7,9) (6,8) (8,10) (11,13) (13,15) (15,17)
(17,19) (17,21) (19,23) (21,23) (12,14) (14,16)
(16,18) (18,20) (18,22) (20,24) (22,24)
```

Usar estas arestas apenas como ponto de partida para visualização. Rever pelo menos 100 imagens escolhidas de forma reprodutível e distribuídas por intervalos do dataset, incluindo exemplos com pés junto ao limite inferior. A revisão deve verificar lado anatómico, localização, ligações dos pés e transformações de resize/flip. Se o skeleton bruto estiver anatomicamente errado, propor uma versão corrigida em `docs/dataset_spec.md`, com justificação e exemplos, antes de o usar em visualizações finais. Não alterar silenciosamente a ordem dos pontos.

### 4.2 Validação e reparação derivada

`validate_dataset` verifica IDs únicos e referenciados, 92 824 ficheiros presentes, dimensões, categoria, comprimento 90 de `keypoints`, valores de visibilidade, `num_keypoints`, finitude das coordenadas e validade das bounding boxes. O relatório separa erros inesperados das 18 boxes e 84 pontos fora da imagem já conhecidos; um aumento desses números exige investigação. Verificar também que o ficheiro de origem não mudou face ao hash registado.

`prepare_splits` não reescreve o JSON original. Para cada box fora da imagem, calcular `(x0,y0,x1,y1)` a partir de `bbox`, limitar às dimensões `[0,width] × [0,height]`, reconstruir `(x,y,w,h)` e atualizar `area` para `w × h`. Para cada ponto visível fora da área de píxeis, definir `(x,y,v)=(0,0,0)` **apenas na cópia derivada** e recalcular `num_keypoints`; não projetar o ponto na borda. Gravar `image_id`, `annotation_id`, valor original, valor derivado e motivo em `data/srkd/derived/corrections.csv`. O validador volta a correr sobre os três JSONs produzidos; nenhuma box inválida nem ponto visível fora da imagem pode restar.

### 4.3 Grupos reconstruídos e auditoria

O manifesto contém `image_id,file_name,group_id,group_method,audit_status` para todas as imagens. A construção inicial segmenta intervalos contíguos de nomes por mudanças visuais de cenário/câmara e encontra candidatos a duplicação ou mesma sequência por semelhança visual entre intervalos. Proximidade numérica, sozinha, nunca basta para declarar independência. Fixar no código os parâmetros e a seed usados; guardar os candidatos e evidência visual para revisão. A atribuição provisória do split é um artefacto separado, descartado sempre que os grupos mudarem.

Auditar manualmente pelo menos 100 pares em limites de grupos e os 100 pares mais semelhantes que cairiam em splits diferentes, além de todos os casos sinalizados como ambíguos pela rotina de agrupamento. Se um par representar a mesma sequência ou cenário indistinguível, reunir os grupos e repetir atribuição e auditoria. Candidatos sem decisão ficam `pending` e bloqueiam o split oficial. O relatório `docs/split_audit.md` identifica método, amostra revista, decisões, limitações e a conclusão **«sem sobreposição conhecida após auditoria»**, nunca uma garantia absoluta de ausência de leakage.

Aplicar 80/10/10 por `group_id`, com seed 42, sem dividir um grupo. Balancear aproximadamente número de imagens e a cobertura de cenários/poses observáveis, sem mover imagens individuais entre splits. O teste é selado antes do treino: não o usar para escolher hiperparâmetros, augmentations, checkpoint ou thresholds. Gerar `train.json`, `val.json`, `test.json` e relatório com contagens de imagens, grupos, visibilidade e condições observáveis. Se um conjunto perder uma condição relevante identificada na auditoria, refazer a atribuição de grupos e registar a nova versão.

**Gate M0:** validação derivada sem erros; 30 pontos e flip/skeleton verificados; 100 overlays revistos; manifesto completo sem `pending`; interseções de `group_id` vazias; auditoria assinada no relatório; hash dos inputs e outputs guardado. Sem este gate, só são permitidos testes locais de integração e overfit, não treino completo ou métricas finais.

## 5. M1 — ambiente e primeiro modelo

### 5.1 Ambiente e configuração

Criar ambiente Python 3.11 dedicado ao PosturaAI, separado da `.venv` Flask/Python 3.13. Instalar PyTorch para o driver/CUDA efetivamente disponível na GPU local; depois MMEngine, MMCV 2.x, MMDetection 3.x e MMPose 1.x compatíveis. Antes do treino, registar `nvidia-smi`, `torch.cuda.is_available()`, versões, importações e um forward pass CPU/GPU. Se a combinação não funcionar, resolver e fixar versões em `requirements-lock.txt`; não iniciar epochs com importações ou operadores CUDA em falha. Seguir a [matriz e ordem de instalação MMPose](https://github.com/open-mmlab/mmpose/blob/main/docs/en/installation.md).

Partir da [configuração oficial RTMPose-S COCO 256×192](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py): CSPNeXt pré-treinado, `RTMCCHead.out_channels=30`, codec SimCC e `CocoDataset` com metainfo SRKD. Carregar somente os pesos compatíveis do backbone; inicializar de novo as camadas da cabeça dependentes dos 17 pontos COCO. Confirmar quais chaves foram carregadas/ignoradas e guardar a lista no log. O [parser de metainfo MMPose](https://github.com/open-mmlab/mmpose/blob/main/mmpose/datasets/datasets/utils.py) exige a chave `sigmas`; até existir calibração, fornecer 30 zeros como **sentinela técnica**, marcar `sigmas_calibrated=False` e proibir `CocoMetric`/AP/OKS na configuração e nos relatórios. Esses zeros não são estimativas estatísticas nem podem ser usados para calcular OKS. Não herdar os 17 sigmas COCO.

Usar batch físico inicial 16, reduzindo para 8, 4 ou 2 em caso de falta de memória; acumular gradientes até batch efetivo 64. Fixar seed 42 e ajustar workers ao hardware. Manter a resolução 256×192 e o mesmo protocolo de validação nas três etapas. Registar batch físico, acumulação, número de workers e GPU reais em cada run.

### 5.2 Treino em etapas

| Etapa | Máximo | Parâmetros treináveis | LR inicial | Paragem antecipada |
| --- | ---: | --- | --- | ---: |
| Smoke | 1–3 épocas em 64 imagens fixas | Backbone e cabeça | LR reduzida de depuração | Não se aplica |
| 1 — cabeça | 10 épocas | Cabeça; backbone congelado, incluindo estatísticas BN | Cabeça `1e-4` | 4 validações sem melhoria |
| 2 — parcial | 25 épocas adicionais | Blocos finais do CSPNeXt e cabeça | Backbone `1e-5`; cabeça `1e-4` | 8 validações sem melhoria |
| 3 — global | 25 épocas adicionais | Backbone completo e cabeça | Backbone `5e-6`; cabeça `2e-5` | 10 validações sem melhoria |

Usar AdamW com weight decay `0.05`, warmup de até 500 iterações de treino e scheduler cosseno por etapa. Smoke deve mostrar gradientes finitos, perda claramente descendente e previsões alinhadas nas 64 imagens; perda estática ou NaN bloqueia a etapa 1. Na etapa 1, usar flip com pares SRKD, pequenas rotações, escala e alterações moderadas de luz. Na etapa 2, acrescentar blur, compressão JPEG e oclusão parcial controlada; qualquer crop deve atualizar visibilidade e preservar supervisão útil dos pés. Na etapa 3, manter apenas augmentations aprovadas pela validação.

Validar pelo menos no fim de cada época e guardar `best.pth` pelo menor NME global de validação; guardar também `latest.pth` e um relatório do melhor erro dos pés, sem usá-lo para substituir automaticamente o critério primário. Selecionar o melhor candidato entre etapas 1–3 **somente com validação**. A etapa 1 exige NME de validação menor que na sua primeira época, sem colapso visual. A etapa 2 avança se melhorar NME global ou NME dos pés pelo menos 1% face à etapa 1, sem piorar a outra mais de 5%; caso contrário, rever LR/augmentations e repetir a etapa 2 uma vez. Se a repetição também falhar, parar M1 e investigar antes de qualquer teste final. A etapa 3 só substitui o melhor checkpoint anterior se reduzir NME global sem piorar NME dos pés mais de 5%; se não o fizer, conservar o checkpoint anterior. Nenhuma etapa usa resultados de `test` para decidir.

Interromper e investigar se houver loss/gradientes não finitos, keypoints colapsados num ponto, número de saídas diferente de 30 ou aumento persistente do erro de validação. Guardar a configuração efetiva, commit, seed, versões, hardware, URL/hash do peso inicial, estado das camadas congeladas, métricas por época e caminho do checkpoint em `outputs/<run_id>/run_metadata.json`.

## 6. Avaliação e artefactos de M1

Definir, para cada ponto visível, `erro_px = distância_euclidiana(pred, gt)` e `erro_normalizado = erro_px / max(altura_bbox_gt_derivada, 1)`. **NME global** é a média dos erros normalizados dos pontos visíveis; **PCK@0,05** é a fração com erro normalizado ≤ 0,05. Calcular ambos por ponto e para o grupo dos pés `15–24`, além de média, mediana e P90 do erro dos pés em píxeis e normalizado. Reportar número de observações por métrica. Não usar `confidence` como substituto de visibilidade do ground truth.

AP/AP50/AP75/AR/OKS ficam **não reportados** até existir um vetor de 30 sigmas justificado e versionado, preferencialmente estimado de repetição de anotação ou outra base empírica documentada. Não preencher com sigmas COCO copiados para pontos novos. Quando disponíveis, esses indicadores passam a métricas adicionais, sem reescrever resultados anteriores.

Após escolher o checkpoint final na validação, avaliar `test` uma única vez e produzir `docs/evaluation_report.md`, tabelas por ponto e pés, métricas globais, configuração/hash do modelo, descrição do split inferido e risco residual. Guardar exemplos visuais bons e maus, incluindo pelo menos 20 casos de maior erro nos pés. Se houver defeito de dados descoberto no teste, versionar nova preparação e tratar qualquer nova medição como nova experiência, nunca como ajuste da experiência já testada.

**Artefactos obrigatórios de M1:** `data/srkd/group_manifest.csv`, três JSONs COCO derivados, `corrections.csv`, `docs/dataset_spec.md`, `docs/split_audit.md`, configs efetivas, logs, `outputs/final/best.pth`, `run_metadata.json`, relatório de teste e visualizações. `outputs/final/best.pth` é uma cópia identificada do melhor checkpoint de validação; o relatório indica a etapa de origem.

## 7. Marcos posteriores

**M2 — imagem e vídeo real.** Recolher 50–200 imagens e vídeos curtos com ângulos, luz e roupa variados, com origem e consentimento documentados. Primeiro medir qualitativamente o sim-to-real gap; métricas quantitativas exigem anotações reais. Para vídeo: descodificar frames → detetar pessoas → RTMPose top-down → associar `track_id` por IoU → suavizar por track com EMA como baseline → renderizar e exportar. Medir frames sem deteção, keypoints em falta, jitter, variância de confiança e latência. Não suavizar através de descontinuidades de track. O contrato de exportação por pessoa é `T × 30 × 3` (`x`, `y`, confiança), com `video_id`, `fps`, `frame`, `track_id`, bbox e indicação de pontos em falta; vídeo anotado e JSON devem ser gerados pelo mesmo comando sem reexecutar o modelo.

**M3 — análise biomecânica.** Consumir apenas o JSON de keypoints, sem dependência direta de MMPose. Normalizar pela pelvis e por uma escala corporal documentada; considerar direção de corrida e pontos ausentes. Implementar ângulos de anca, joelho, tornozelo, tronco, cotovelo e ombro; relações espaciais e sinais temporais de passada. Cada feature especifica fórmula, unidade, pontos, tratamento de baixa confiança e teste. Regras para overstriding, inclinação do tronco, hip drop e oscilação vertical devolvem score contínuo, classificação e evidência; thresholds só são fixados depois de validar exemplos.

**M4 — classificação temporal.** Só criar treino supervisionado quando houver sequências de 1–3 segundos com rótulos multi-label revistos e split por corredor/sequência. Começar por MLP de features agregadas; usar TCN ou ST-GCN apenas se melhorar a validação. Medir precision, recall, F1 e PR-AUC por classe. Um conjunto sem rótulos validados não autoriza métricas de classificação.

**M5 — Flask.** Integrar upload/processamento na aplicação RunningAI depois de estabilizar M2/M3, com execução desacoplada do pedido HTTP para vídeos longos. Mostrar artefactos e indicadores já produzidos pelo pipeline; não treinar modelos dentro da aplicação web.

## 8. Testes e definição de pronto da primeira entrega

Testes automáticos devem cobrir: integridade e paths do SRKD; correções geométricas sem mutação do original; 30 pontos, pares de flip e índices do skeleton; manifesto completo e interseção vazia de grupos entre splits; determinismo com seed 42; forward pass e recarregamento do checkpoint com shape `30 × 2` de coordenadas e 30 confidences; métricas em exemplos calculáveis à mão; e inferência top-down numa imagem SRKD. Uma auditoria visual de 100 overlays e dos pares de grupos definidos em 4.3 complementa os testes.

M1 fica concluído apenas quando: gate M0 passou; ambiente GPU e versões foram registados; smoke reduziu loss; as três etapas foram executadas ou interrompidas pelos critérios documentados; o melhor checkpoint foi escolhido em `val`; NME/PCK e erro dos pés foram medidos em `test` uma vez; existe relatório com casos de falha e risco residual de leakage; e outro agente consegue reproduzir preparação e treino a partir de comandos, configurações e hashes guardados. A existência de `best.pth` sem estes artefactos não conclui o marco.
