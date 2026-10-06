# Documentação PosturaAI

| Tema | Documentos |
| --- | --- |
| Desenvolvimento | [Plano de implementação](plans/rtmpose_srk_implementation_plan.md), [proposta de análise com LLM local](plans/local_llm_feedback.md) |
| Dataset sintético | [Especificação](data/dataset_spec.md), [preparação M0](data/m0_preparation.md), [revisão visual](data/visual_audit.md), [grupos](data/split_audit.md) |
| Treino | [Ambiente](training/training_environment.md), [relatório inicial](training/first_training_report.md) |
| Vídeo | [Pipeline e comandos](video/video_pipeline.md), [resultados da execução](video/video_pipeline_report.md), [primeiro classificador](video/classifier.md) |
| Vídeos rotulados | [Instruções das pastas](../data/running_posture/README.md), [catálogo](../data/running_posture/video_manifest.csv) |

O estado atual e os comandos principais estão no [README do projeto](../README.md).
Os relatórios históricos identificam a fase em que foram produzidos.

Os caminhos anteriores à reorganização estão mapeados em
[structure_migration.json](structure_migration.json). Os resultados antigos
ficam em `outputs/archive/`; o modelo de referência está em
`outputs/stage1_experimental/best.pth`.
