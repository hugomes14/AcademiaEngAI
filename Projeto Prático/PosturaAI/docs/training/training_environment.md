# Ambiente de treino PosturaAI

## Hardware verificado em 2026-10-06

`nvidia-smi`, executado fora do sandbox, detetou uma NVIDIA GeForce RTX 5080 com 16 303 MiB, driver NVIDIA-SMI 610.55 / KMD 610.78 e CUDA UMD 13.3. Dentro do sandbox, o sistema operativo bloqueia o acesso à GPU. Os comandos CUDA desta sessão precisam de execução fora do sandbox; o terminal normal do utilizador não tem essa restrição específica do agente.

Esta verificação corrige a conclusão anterior de que a GPU não estava disponível no sistema. PyTorch executou uma operação matricial na GPU, o forward RTMPose produziu 30 canais finitos por eixo SimCC, o smoke concluiu três épocas e a inferência recarregou o checkpoint. Consultar [relatório do primeiro treino](first_training_report.md).

## Instalação isolada

O Python 3.11.17 foi descarregado para `.runtime/python/` e a `.venv` pertence exclusivamente à PosturaAI. A aplicação Flask mantém o seu ambiente separado. `.runtime/`, `.venv/` e os pesos em `outputs/` são ignorados pelo Git.

A [documentação PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) introduz suporte Blackwell e wheels CUDA 12.8. Foi selecionado PyTorch 2.7.1 / torchvision 0.22.1 com CUDA 12.8. As versões OpenMMLab seguem a família [MMPose 1.x / MMDetection 3.x / MMCV 2.x](https://github.com/open-mmlab/mmpose/blob/v1.3.2/docs/en/installation.md).

Comandos a partir da raiz PosturaAI, com uma `.venv` Python 3.11 existente:

```bash
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install setuptools==80.9.0 wheel ninja psutil
.venv/bin/python -m pip install --no-build-isolation -r requirements-training.txt
git clone --depth 1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git .runtime/mmcv
CUDA_VISIBLE_DEVICES='' MMCV_WITH_OPS=1 MAX_JOBS=2 .venv/bin/python -m pip install --no-build-isolation --no-deps .runtime/mmcv
```

O MMCV é compilado a partir do código oficial v2.1.0, commit `57c4e25e06e2d4f8a9357c84bcd24089a284dc88`, com operadores CPU. A [documentação de compilação MMCV](https://mmcv.readthedocs.io/en/2.x/get_started/build.html) distingue operadores CPU e CUDA. O RTMPose-S usa operações PyTorch na GPU; a ausência de operadores MMCV CUDA terá de ser considerada ao integrar outros modelos que os exijam, como um detetor de pessoas em M2. Não foi instalado um toolkit `nvcc` no sistema.

O backbone oficial foi descarregado para `outputs/pretrained/cspnext_s_aic_coco.pth`:

- URL: <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth>
- SHA-256: `aa7d9335bf422ad02a803e36f357dfc6abb807eca42d79e8b3b6e7c5bd1f446b`.

Usar o ficheiro local para conservar esse hash nos metadados de treino:

```bash
.venv/bin/python -m scripts.train --stage smoke --run-id NOVO_RUN_SMOKE --backbone-checkpoint outputs/pretrained/cspnext_s_aic_coco.pth
.venv/bin/python -m scripts.train --stage 1 --allow-unaudited --run-id stage1_experimental --backbone-checkpoint outputs/pretrained/cspnext_s_aic_coco.pth
```

Os resultados pertencem à experiência sem auditoria autorizada pelo utilizador. O relatório da preparação contém a proveniência dos splits; `run_metadata.json` conserva o estado experimental, versões, hardware, configuração e hashes.

O checkpoint oficial contém metadados NumPy que não são aceites por omissão no carregamento restrito de PyTorch 2.7. `checkpoint_load_context` permite explicitamente os tipos NumPy e `HistoryBuffer` usados nesses metadados, apenas durante a execução do treino. O carregamento restrito mantém-se ativo.

`preflight_runtime` define `CUBLAS_WORKSPACE_CONFIG=:4096:8` antes de inicializar CUDA, necessário para o determinismo configurado. A configuração efetiva fica nos metadados de cada execução. As versões completas resolvidas estão em `requirements-lock.txt`; o build MMCV usa o source e a ordem de instalação acima.
