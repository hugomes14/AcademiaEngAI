# SRKD: especificação dos keypoints e auditoria visual

**Estado:** metadados transcritos do JSON bruto; skeleton e lateralidade ainda aguardam revisão visual M0. Esta especificação não constitui aprovação do gate M0.

## Origem e convenções

O ficheiro de referência é `Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/keypoints_srkd.json`, categoria `person` (`id=1`). Há uma pessoa por imagem. Cada anotação guarda 30 triplos `(x, y, v)` em píxeis da imagem original 640 × 360; `v=0` significa ausente e `v>0` significa ponto utilizável. O JSON bruto usa `v=2` para todos os pontos, incluindo 84 pontos dos pés fora da imagem. O tratamento desses pontos ocorre somente nas anotações derivadas, conforme o plano de implementação.

Os índices desta tabela e do código são **0-based**. As ligações de `categories[0].skeleton` no JSON original são **1-based**. `src/posturaai/keypoints.py` guarda a ordem original e as 33 ligações únicas já convertidas.

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

O `horizontal flip` troca `(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16), (17,18), (19,20), (21,22), (23,24)`. Os pontos `0,25,26,27,28,29` mantêm o canal. A permutação aplicada duas vezes deve devolver a ordem original.

As ligações provisórias do skeleton são:

```text
(0,1) (0,2) (1,3) (2,4) (25,26) (0,26) (0,27) (5,27)
(6,27) (27,28) (0,3) (1,2) (28,29) (11,29) (12,29)
(5,7) (7,9) (6,8) (8,10) (11,13) (13,15) (15,17)
(17,19) (17,21) (19,23) (21,23) (12,14) (14,16)
(16,18) (18,20) (18,22) (20,24) (22,24)
```

O JSON original repete duas ligações. O código normaliza cada par como ligação não dirigida (menor índice primeiro), remove as repetições e mantém a ordem da primeira ocorrência. A metainfo para MMPose fornece `keypoint_info`, `skeleton_info`, pesos unitários e 30 valores `sigma=0` exigidos pelo parser, marcados `sigmas_calibrated=False`. Estes zeros são apenas sentinelas de compatibilidade, **não uma calibração**: AP/OKS tem de permanecer desativado até serem justificados 30 sigmas próprios do SRKD.

## Inspeção visual M0

Executar, a partir da raiz PosturaAI:

```bash
python -m scripts.visualize_annotations --samples 100 --output outputs/visualizations/ground_truth
```

O comando usa seed 42 e sorteia imagens por dez intervalos de ID, reservando em cada intervalo um caso com pés até 12 píxeis do limite inferior, se existir. A pasta de saída contém os overlays JPEG e `sample_list.csv` com IDs, caminho da imagem original, motivo da seleção e indicação de box/pontos fora da imagem. Os pontos ciano representam o lado esquerdo, os laranja o direito e os brancos pontos centrais; as arestas verdes são provisórias e a box é magenta. Cada ponto visível tem o índice escrito ao lado.

Rever os **100 overlays** e registar no relatório de auditoria: IDs revistos, discrepâncias de lado anatómico, localização e ligação dos pés, erros de coordenadas, efeitos de resize/flip e decisão sobre manter ou corrigir o skeleton. Rever especificamente os casos `foot_near_lower_edge` e os marcados com pontos fora da imagem. Se for necessária uma correção de skeleton, documentar os pares originais e propostos, a justificação e os IDs de exemplo antes de alterar as visualizações finais. Não alterar a ordem dos 30 pontos nem o JSON bruto.

**Revisão humana:** pendente. Nenhuma ligação foi aprovada nesta fase.
