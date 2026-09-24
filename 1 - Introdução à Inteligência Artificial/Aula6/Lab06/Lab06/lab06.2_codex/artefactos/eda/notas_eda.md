# Notas da Análise Exploratória

- O dataset tem **1000 linhas** e **6 colunas**.
- A variável-alvo é `incidente_reportado`.
- A classe positiva (`1`, incidente) representa **10.00%** dos registos.
- Existe desbalanceamento de classes; por isso, a Accuracy deve ser interpretada com cuidado.
- Em problemas de segurança, falsos negativos são críticos: voos de risco classificados como seguros.
- O pré-processamento deve ajustar codificadores e escalonadores apenas no treino para evitar data leakage.
- O escalonamento é importante para KNN e SVM porque estes modelos dependem de distâncias ou margens.