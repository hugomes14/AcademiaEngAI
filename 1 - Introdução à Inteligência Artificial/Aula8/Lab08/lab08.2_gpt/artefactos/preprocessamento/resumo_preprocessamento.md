# Resumo do Pré-processamento

- Dataset: `dados_voo_ga.csv`
- Target: `incidente_reportado`
- Número de features: 18
- Linhas treino: 450
- Linhas teste: 150
- Split: estratificado
- Escalonamento: `StandardScaler`

## Nota sobre data leakage

O `StandardScaler` foi ajustado apenas no conjunto de treino. Depois, o mesmo transformador foi aplicado ao teste. Isto evita que estatísticas do teste influenciem a preparação dos dados.
