# Notas da Análise de Resíduos

## voltagem
- Melhor modelo: `Regressao_Linear`.
- Média dos resíduos: -0.0004.
- Desvio-padrão dos resíduos: 0.0505.
- Autocorrelação lag 1 dos resíduos: 0.0110.
- Interpretação: resíduos com média perto de zero indicam baixo viés global.
- Se a ACF dos resíduos tiver valores relevantes, o modelo ainda deixou estrutura temporal por capturar.

## missoes
- Melhor modelo: `Ridge`.
- Média dos resíduos: 0.1396.
- Desvio-padrão dos resíduos: 3.3884.
- Autocorrelação lag 1 dos resíduos: -0.0960.
- Interpretação: resíduos com média perto de zero indicam baixo viés global.
- Se a ACF dos resíduos tiver valores relevantes, o modelo ainda deixou estrutura temporal por capturar.
