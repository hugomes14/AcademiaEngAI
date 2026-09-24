# PosturaAI

Projeto de estimação de pose para corrida da plataforma RunningAI. O objetivo da primeira entrega é treinar e validar um RTMPose com os 30 keypoints do Synthetic Runner Keypoint Dataset (SRKD). Vídeo, biomecânica e integração Flask são marcos posteriores.

O [plano de implementação](rtmpose_srk_implementation_plan.md) é a fonte de verdade para arquitetura, critérios de avanço e artefactos esperados. O trabalho começou pelo marco M0: integridade do dataset, documentação dos pontos, visualização e preparação de splits auditados. Ainda não existe modelo treinado.

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

Depois de rever as visualizações, consultar a sequência de construção e auditoria dos grupos no plano antes de gerar `train`, `val` e `test`. O treino RTMPose depende de uma GPU local e de um conjunto compatível PyTorch/CUDA/OpenMMLab, ainda por fixar.
