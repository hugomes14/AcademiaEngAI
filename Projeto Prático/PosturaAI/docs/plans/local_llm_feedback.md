# LLM local para explicar previsões e comentar a corrida

Registado em **2026-10-06**, a pedido do utilizador. **Estado: proposta guardada; integração ainda não implementada.** O utilizador tem LM Studio e Unsloth; os modelos instalados e o endereço do servidor ainda não foram identificados.

## Objetivo

Acrescentar à app uma análise em português que explique as previsões de boa/má postura e, quando existir evidência suficiente, sugira aspetos a rever na corrida. Manter a experiência atual: o utilizador importa um vídeo e acompanha o processamento; o relatório surge na mesma página.

## Contexto atual

- A app principal já disponibiliza `/postura-ai`, com upload, fila em segundo plano, progresso, pré-visualização da pose e reprodução/descarga do vídeo anotado.
- RTMPose estima 30 pontos; o pipeline acompanha pessoas e extrai medidas projetadas em 2D.
- O classificador usa 35 descritores agregados de joelho, anca, tornozelo, cotovelo, ombro, inclinação do tronco e relação cabeça/tronco. Após esta proposta, a app passou de regressão logística para Extra Trees; ver [comparação de modelos](../training/nonlinear_classifier_report.md).
- A validação por grupo do primeiro classificador foi fraca: cerca de 52% de acurácia e **39,12% de acurácia equilibrada**. Os scores não são probabilidades calibradas; o vídeo de teste não tem ground truth de postura.
- Os vídeos devem abranger vários planos. Não exigir ao utilizador que controle ou indique o plano; conservar as limitações da perspetiva na interpretação das medidas.

O LLM não melhora, por si só, a precisão do classificador. Uma explicação fluente pode justificar uma previsão errada; o relatório deve conservar a incerteza.

## Abordagem proposta

Fluxo: **vídeo → pose e medidas → classificação e contribuições → LLM local → relatório na app**.

Começar por um modelo já instalado no **LM Studio**, através da API local. Não é necessário ajustar um LLM para a primeira versão. Pedir uma resposta estruturada e validá-la antes de a apresentar. Manter URL e modelo configuráveis no servidor, sem escolhas técnicas adicionais no formulário de vídeo.

Reservar o **Unsloth** para uma fase posterior de ajuste com exemplos de análises revistos por um treinador ou especialista. Esses exemplos devem ligar medidas, limitações, timestamps e comentários, em vez de ensinar o modelo a repetir os rótulos experimentais como verdade.

Gerar o relatório depois da inferência de pose para reduzir concorrência pela GPU. O hardware conhecido é uma RTX 5080 Laptop com 16 GB de VRAM; o modelo e a quantização devem ser escolhidos após verificar o que está instalado e medir o consumo real.

## Explicação da decisão e sugestões

**Explicação da previsão:** calcular no código as contribuições dos descritores para o logit da regressão logística, após a mesma imputação e normalização usadas na inferência. Entregar ao LLM os fatores que mais favoreceram cada classe. Estes fatores explicam o cálculo do modelo; não estabelecem causalidade biomecânica. Identificar descritores imputados, pois não são observações do vídeo.

**Atualização após a troca para Extra Trees:** a abordagem com coeficientes/logit
refere-se ao modelo anterior. Para o modelo ativo, definir e validar atribuições
específicas de árvores antes de apresentar explicações individuais. A importância
global das features não deve ser apresentada como justificação de uma decisão
concreta. Conservar as observações e a incerteza mesmo sem atribuições disponíveis.

**Sugestões de melhoria:** apoiar os comentários numa base de conhecimento de técnica de corrida, com fontes e critérios revistos por alguém da área. Uma medida que contribuiu para “má postura” não demonstra que exista um erro técnico. Não inventar ângulos ideais nem correções com base apenas no rótulo.

## Informação a entregar ao LLM

Resumir por sequência, sem enviar todos os frames na primeira versão:

- Intervalo temporal e pessoa/segmento acompanhados.
- Medidas observadas, unidades e variação temporal.
- Qualidade dos pontos, dados ausentes e medidas imputadas.
- Previsão experimental, score não calibrado e principais contribuições calculadas.
- Plano/direção/câmara quando conhecidos; indicar explicitamente quando forem desconhecidos.
- Limitações das medidas projetadas e excertos relevantes da base de conhecimento, quando disponível.

O LLM deve distinguir observações, interpretação do classificador e sugestões. Quando a evidência não permitir avaliar um aspeto, deve dizê-lo. Ângulos projetados em 2D não equivalem automaticamente a medidas biomecânicas reais e não devem ser comparados indiscriminadamente entre planos.

## Apresentação na app

Adicionar uma área **“Análise comentada”** com:

- Observações ligadas a timestamps do vídeo.
- Explicação das previsões e dos fatores que as influenciaram.
- Aspetos a rever e sugestões sustentadas em evidência disponível.
- Limitações e indicação “Não é possível avaliar” quando necessário.

Mostrar o estado de geração do relatório. Se o servidor LLM estiver desligado, indisponível ou devolver uma resposta inválida, conservar o vídeo, as medidas e a classificação; apresentar a indisponibilidade apenas no relatório.

## Próximos passos quando a implementação for solicitada

1. Identificar os modelos instalados no LM Studio e a conectividade a partir do processo Flask (pode estar noutro sistema, por exemplo Windows/WSL).
2. Definir o contrato do resumo de evidências, das contribuições e da resposta estruturada.
3. Implementar o adaptador local com timeout, validação e tratamento de indisponibilidade.
4. Integrar a geração no job e apresentar o relatório na página do vídeo.
5. Verificar fidelidade às medidas/contribuições, referências temporais, ausência de valores inventados e comportamento perante dados insuficientes ou planos desconhecidos.
6. Reunir exemplos revistos para sugestões de técnica e avaliar, posteriormente, se um ajuste com Unsloth traz benefício.

## Referências consultadas

- [LM Studio: servidor local](https://lmstudio.ai/docs/developer/core/server).
- [LM Studio: endpoints compatíveis com OpenAI](https://lmstudio.ai/docs/developer/openai-compat).
- [LM Studio: resposta estruturada](https://lmstudio.ai/docs/developer/openai-compat/structured-output).
- [Unsloth: guia de ajuste de LLMs](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide).
