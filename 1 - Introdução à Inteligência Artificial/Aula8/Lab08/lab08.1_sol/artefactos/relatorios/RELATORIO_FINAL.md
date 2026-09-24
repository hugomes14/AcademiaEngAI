# Relatório Final — Lab 08.1 Aprendizagem por Reforço

## 1. Introdução

Este laboratório demonstra dois cenários clássicos de Aprendizagem por Reforço: um problema Multi-Armed Bandit para seleção de rotas de voo e um Grid World para aprendizagem de uma política de navegação com Q-Learning.

## 2. Ambiente e Agente

- **Bandit:** cada rota representa uma ação com recompensa incerta. O agente procura maximizar a recompensa ao longo do tempo.
- **Grid World:** cada estado representa uma posição numa grelha 10x10. O agente aprende uma política que conduz ao destino e evita obstáculos.
- **Q-Learning:** usa uma Q-table para estimar o valor de cada par estado-ação.

## 3. Exploração vs. Explotação no Bandit

| algoritmo         |   recompensa_media_final |   regret_acumulado_final |
|:------------------|-------------------------:|-------------------------:|
| thompson_sampling |                  1.87942 |                   51.45  |
| ucb               |                  1.84306 |                  142.34  |
| epsilon_greedy    |                  1.81992 |                  200.189 |


![Recompensa média acumulada](../imagens/bandit_recompensa_media_acumulada.png)

![Regret acumulado](../imagens/bandit_regret_acumulado.png)

O regret acumulado mede a diferença entre a recompensa obtida e a recompensa que seria esperada se o agente escolhesse sempre a melhor rota real. Um regret mais baixo indica aprendizagem mais eficiente.

## 4. Treino com Q-Learning

O treino usa uma política epsilon-greedy. No início, `epsilon` é alto para favorecer exploração. Ao longo dos episódios, `epsilon` desce para favorecer a exploração da melhor política conhecida.

- Recompensa média móvel final: **7.676**

- Taxa de sucesso móvel final: **97.00%**

![Curva de aprendizagem](../imagens/qlearning_curva_aprendizagem.png)

![Taxa de sucesso](../imagens/qlearning_taxa_sucesso.png)

![Decaimento de epsilon](../imagens/qlearning_epsilon.png)

## 5. Avaliação da Política

| agente           |   episodios |   recompensa_media |   recompensa_desvio_padrao |   taxa_sucesso |   passos_medios |
|:-----------------|------------:|-------------------:|---------------------------:|---------------:|----------------:|
| agente_treinado  |         300 |              8.3   |                    0       |              1 |           18    |
| agente_aleatorio |         300 |            -12.562 |                    1.95183 |              0 |           26.62 |


A política treinada deve superar o agente aleatório em recompensa média e taxa de sucesso. A avaliação não usa exploração intencional; por isso, mede a qualidade da política aprendida.

## 6. Visualização da Política

![Função valor](../imagens/qlearning_funcao_valor_heatmap.png)

![Política ótima](../imagens/qlearning_politica_setas.png)

A função valor mostra os estados com maior valor estimado. A política com setas mostra a ação preferida em cada célula normal. Obstáculos e destino aparecem assinalados no mapa.

## 7. Conceitos-Chave

- **Estado:** situação atual do agente.
- **Ação:** decisão possível em cada estado.
- **Recompensa:** sinal que orienta a aprendizagem.
- **Alpha:** controla quanto a nova informação altera a Q-table.
- **Gamma:** controla a importância de recompensas futuras.
- **Epsilon:** controla o equilíbrio entre exploração e explotação.
- **Equação de Bellman:** atualiza o valor esperado de cada ação em cada estado.

## 8. Conclusões e Recomendações

A Aprendizagem por Reforço é adequada para problemas de decisão sequencial, nos quais o efeito de uma ação só se torna claro depois de vários passos. No Bandit, o foco está na seleção repetida da melhor rota. No Grid World, o foco está na aprendizagem de uma política completa de navegação.

Recomendações para aprofundamento:
- Testar diferentes recompensas para analisar *reward shaping*.
- Alterar obstáculos no Grid World para avaliar robustez.
- Comparar Q-Learning com SARSA.
- Explorar DQN quando o espaço de estados deixa de caber numa Q-table.
