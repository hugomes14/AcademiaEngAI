# Preparação da revisão M0 — 2026-10-06

**Registo da preparação anterior à exportação experimental.** A revisão humana permanece pendente; posteriormente, em 2026-10-06, o utilizador decidiu avançar sem auditoria. O modo experimental e os comandos estão no [README](../../README.md). Este registo não aprova os grupos, a lateralidade ou o skeleton.

## Verificações concluídas

- Validação completa: 92 824 imagens presentes, dimensões 640 × 360, uma anotação por imagem, sem erros inesperados. Mantêm-se as 18 boxes e 84 pontos fora da imagem já conhecidos.
- 16 testes passaram. A configuração da etapa 1 pode ser resolvida com `python -m scripts.train --stage 1 --show-config`; o runtime de treino continua por instalar e verificar.
- Manifesto: 153 grupos, todos `pending`. A distribuição com seed 42 é reproduzível e cada grupo pertence a um único split.
- Auditoria preparada: 101 pares de limites, 100 entre splits e 4 candidatos visuais adicionais; 9 dos 205 pares estão marcados como ambíguos. Todas as decisões e os nomes dos revisores estão vazios.
- Evidência: 100 overlays, 21 folhas de pares e uma galeria com 305 entradas. As 610 referências de imagens da galeria foram verificadas.

| Split provisório | Imagens | Grupos |
| --- | ---: | ---: |
| Treino | 74 278 | 108 |
| Validação | 9 248 | 22 |
| Teste | 9 298 | 23 |

A distribuição é provisória: as imagens ainda não foram exportadas para os JSONs finais. Não existe `audit_approval.json`.

## Prioridade de revisão

A inspeção inicial de `outputs/visualizations/group_audit/pairs_012.jpg` sugere que os grupos `g037389` e `g078263` partilham um cenário noturno muito semelhante, com árvores, iluminação e enquadramento próximos. Exemplo: imagens 37909 e 78373, candidato `4175a93666c2ceb9`. Os cinco primeiros pares dessa folha comparam esses dois grupos.

Os cinco pares seguintes comparam `g036945` e `g078263`, com aparente mudança de enquadramento no mesmo tipo de cenário. Exemplo: imagens 37061 e 78559, candidato `6acc0557aea19f3d`.

São observações preliminares feitas pelo assistente, não decisões assinadas. Confirmar nas imagens originais se são a mesma sequência ou cenário indistinguível; nesse caso, registar `same`, reunir os grupos e repetir a atribuição e auditoria. Não usar semelhança de cenário como prova de identidade do corredor.

## Evidência e proveniência

Abrir `outputs/visualizations/m0_review.html` no navegador. O comando `python -m scripts.build_audit_gallery` regenera a galeria a partir dos CSVs existentes, sem modificar decisões ou anotações. A galeria não verifica resize/flip; essa verificação continua a fazer parte da revisão anatómica M0.

Os hashes abaixo identificam esta preparação **antes de qualquer decisão ou fusão**:

| Artefacto | SHA-256 |
| --- | --- |
| JSON bruto SRKD | `66c8b3927c238ffd0efe878d7854b494a5afc95e6acf2b1a5e73d0cbb9b14da3` |
| `data/srkd/group_manifest.csv` | `9ece1340eb95dc64b285ba48cfae24098a30245fc0d9811fed9608216d9cbba4` |
| `data/srkd/provisional_assignment.csv` | `bfd8599e2df8bfb6a8913d0fcf5c97cff6badf0b1c93483065a6ebeb4599ebef` |
| `data/srkd/audit_decisions.csv` | `5d4cdd4801a89ef86f0a0ce95fe44ad074eb9b24bb60154b0e1899b3f7c57029` |
| `outputs/visualizations/ground_truth/sample_list.csv` | `157e0e4986f3253e5658329371c7f17e78a36ce94db47ac9949df53d43392da6` |

O manifesto e a configuração SRKD podem agora ser versionados. Caches, imagens, distribuição provisória e decisões continuam ignorados pelo Git; preservar uma cópia local desses artefactos ao concluir cada revisão. A sequência de comandos está no [README](../../README.md); os critérios completos estão no [plano de implementação](../plans/rtmpose_srk_implementation_plan.md).
