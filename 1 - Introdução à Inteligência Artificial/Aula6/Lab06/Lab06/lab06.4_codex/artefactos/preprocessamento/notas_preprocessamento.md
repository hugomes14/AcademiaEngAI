# Notas de Pré-processamento

## Encoding

- `previsao_turbulencia` foi codificada de forma ordinal: Baixa < Média < Alta.
- `tipo_missao` foi codificada com one-hot encoding porque não existe ordem natural.

## Escalonamento

As variáveis numéricas foram escalonadas com `StandardScaler`.

Embora árvores e ensembles baseados em árvores não exijam escalonamento, esta etapa
mantém o pipeline consistente e evita problemas se forem adicionados modelos sensíveis
à escala.

## Data Leakage

O `fit` do pré-processador foi aplicado apenas ao conjunto de treino. O conjunto de teste
só recebeu `transform`. Isto evita que informação estatística do teste entre no treino.

## Desbalanceamento

O split foi estratificado para manter proporções semelhantes de classes no treino e no teste.
