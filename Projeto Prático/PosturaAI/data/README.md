# Dados PosturaAI

| Pasta | Conteúdo |
| --- | --- |
| `srkd/` | Manifesto, preparação e splits do dataset sintético |
| `videos/test/` | Vídeos de referência para testar o pipeline |
| `running_posture/boa_postura/` | Vídeos que classificaste como boa postura |
| `running_posture/ma_postura/` | Vídeos que classificaste como má postura |

O catálogo está em [running_posture/video_manifest.csv](running_posture/video_manifest.csv).
As instruções de classificação estão no [README das pastas](running_posture/README.md).

As imagens brutas SRKD conservam as duas árvores originais na raiz do projeto,
usadas pelas anotações e pelos checkpoints. A especificação está em
[dataset_spec.md](../docs/data/dataset_spec.md).

Os vídeos e splits derivados ficam locais; as instruções e o catálogo são
versionáveis. Guardar os resultados do processamento em `outputs/videos/`.
