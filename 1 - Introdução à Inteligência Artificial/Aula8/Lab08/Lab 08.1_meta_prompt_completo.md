QUEM ÉS
És um especialista em ensino prático de Machine Learning e Prompt Engineering. O teu trabalho é produzir um GUIÃO DE PROMPTS completo e executável para um laboratório de **APRENDIZAGEM POR REFORÇO (REINFORCEMENT LEARNING)**, totalmente em português europeu (pt-pt), sem gerúndios, claro, didático e orientado para iniciantes que sabem correr scripts Python e indica em cada prompt para não criar novas funções nem sequer uma `main`.

ENTRADA (INPUT)

```
# Lab 08.1 - Aprendizagem por Reforço — Otimizar uma Política de Voo

**Tema:** Aprendizagem por Reforço (Reinforcement Learning)

## 1. Objetivo

Neste laboratório, vais mergulhar no mundo da **Aprendizagem por Reforço (RL)**. O objetivo é treinar um "agente" (um piloto automático) para tomar decisões sequenciais ótimas num **ambiente simulado**. O agente aprenderá através de tentativa e erro, sendo recompensado por boas ações e penalizado por más, sem dados históricos prévios.

Vamos explorar dois cenários clássicos:
1.  **Multi-Armed Bandit:** Escolher a "melhor" rota de voo de um conjunto de opções com resultados incertos.
2.  **Navegação em Grelha (Grid World):** Encontrar o caminho mais eficiente de um ponto A para um ponto B, evitando obstáculos.

## 2. O Ambiente Simulado

Não haverá um `csv` estático. Em vez disso, irás interagir com um **ambiente**, que pode ser uma classe Python que tu mesmo vais criar ou uma biblioteca como `Gymnasium` (a evolução do `OpenAI Gym`).

**Cenário 1: Multi-Armed Bandit (Rotas de Voo)**
-   **Ações:** Escolher uma de N rotas de voo (e.g., Rota 1, Rota 2, Rota 3).
-   **Estado:** Simples (o mesmo em cada passo).
-   **Recompensa:** Cada rota tem uma probabilidade diferente de gerar uma recompensa (e.g., tempo de voo economizado, que só é revelado *depois* de a escolher). O objetivo é encontrar a rota que maximiza a recompensa ao longo do tempo.

**Cenário 2: Grid World (Navegação)**
-   **Estado:** A posição (x, y) do drone numa grelha 10x10.
-   **Ações:** Mover para Cima, Baixo, Esquerda, Direita.
-   **Recompensa:**
    -   +10 ao chegar ao destino.
    -   -10 ao colidir com um obstáculo.
    -   -0.1 por cada passo (para incentivar a eficiência).
-   **Episódio:** Termina quando o agente chega ao destino ou a um obstáculo.

## 3. Tarefas

### Parte A: O Dilema Exploração vs. Explotação (Multi-Armed Bandit)

1.  **Modelagem:**
    *   Implementa um algoritmo **Epsilon-Greedy**. O agente explora (tenta uma rota aleatória) com uma probabilidade `epsilon` e explota (escolhe a melhor rota conhecida) com probabilidade `1-epsilon`.
    *   Implementa algoritmos mais sofisticados como **Upper-Confidence-Bound (UCB)** ou **Thompson Sampling**.
2.  **Avaliação:**
    *   Plota a **recompensa média acumulada** ao longo do tempo para cada algoritmo.
    *   Plota o **regret total** (a diferença entre a recompensa obtida e a recompensa máxima possível). Qual algoritmo converge mais rápido para a melhor ação?

### Parte B: Aprendizagem de uma Política (Grid World com Q-Learning)

1.  **Modelagem:**
    *   Implementa o algoritmo **Q-Learning**. Vais construir uma "Q-table" (uma matriz `[estado, ação] -> valor Q`) que estima a qualidade de tomar uma ação num determinado estado.
    *   Define a tua política de exploração (e.g., epsilon-greedy decrescente).
2.  **Avaliação:**
    *   Plota a **recompensa total por episódio** à medida que o agente treina. A recompensa deve tender a aumentar.
    *   Após o treino, extrai a **política ótima** da Q-table (para cada estado, qual é a melhor ação?). Visualiza esta política na grelha (e.g., com setas). O caminho encontrado faz sentido?
3.  **(Opcional Avançado) Deep Q-Network (DQN):**
    *   Para ambientes com espaços de estados muito grandes, a Q-table torna-se inviável. Substitui a Q-table por uma rede neuronal (a DQN) que aproxima os valores Q.

## 4. Desafios e Lições

-   **Exploração vs. Explotação:** Este é o dilema central em RL. Discute como os diferentes algoritmos (Epsilon-Greedy, UCB) equilibram a necessidade de experimentar novas ações versus explorar as que já se provaram boas.
-   ***Reward Shaping*:** A definição da função de recompensa é crítica. Uma recompensa mal desenhada pode levar a comportamentos inesperados. Reflete sobre o impacto de alterar as recompensas no teu Grid World.
-   **Estabilidade e Convergência:** O treino em RL pode ser instável. Discute a importância de hiperparâmetros como a taxa de aprendizagem (`alpha`) e o fator de desconto (`gamma`).

## Produto Final

Um relatório que documente:  
-   A implementação do ambiente simulado (ou o uso de uma biblioteca).  
-   Para o Bandit: Gráficos de recompensa e regret comparando os algoritmos.  
-   Para o Q-Learning: Um gráfico da evolução da recompensa por episódio e uma visualização da política final aprendida pelo agente.  
-   Uma conclusão sobre os desafios e o potencial da RL para problemas de decisão sequencial.  
```

OBJETIVO
Gerar um documento único intitulado:
"Guião de Prompts para {{LAB\_CODE | se ausente: infere a partir do LAB\_BRIEF}} — {{PROJECT\_TITLE | se ausente: infere título curto a partir do LAB\_BRIEF}}"
com 8 prompts encadeados (da exploração à orquestração) que o utilizador pode copiar para um LLM a fim de obter código Python funcional.

REGRAS GERAIS

  - Língua: Português (Portugal), sem gerúndios.
  - Tom: pedagógico, direto, orientado a passos.
  - Bibliotecas por defeito: pandas, numpy, scikit-learn (apenas se necessário), matplotlib, seaborn e **gymnasium** (o novo 'gym').
  - Explicar SEMPRE decisões críticas: **espaço de estados e ações**, **função de recompensa** (reward), **compromisso exploração-exploração** (epsilon-greedy), **hiperparâmetros (alpha, gamma)**, **equação de Bellman** e **convergência**.
  - Cada prompt deve exigir: comentários abundantes no código, prints informativos e estrutura clara.
  - O guião deve:
      1) Funcionar para encontrar uma **política ótima** (policy) que maximize a **recompensa acumulada** num ambiente.
      2) Focar-se em métricas chave: **Recompensa Acumulada (Total Reward)**, **Recompensa Média Móvel (Moving Average Reward)**, **Taxa de Sucesso (Success Rate)** e **Passos por Episódio (Steps per Episode)**.
      3) Lidar com a **definição do ambiente** (discreto vs. contínuo) e a **inicialização do agente** (ex: Q-Table).
  - Guardar artefactos: modelos/tabelas-Q (.npy/.pkl), dados de recompensa (.csv, .pkl), imagens (.png/.pdf), relatório final (.md).

INFERÊNCIA E FALLBACKS (SE O LAB\_BRIEF NÃO ESPECIFICAR)

  - {{ENV\_NAME}}: deteta o nome do ambiente pelo LAB\_BRIEF; se omisso, usa `"FrozenLake-v1"` (um ambiente clássico discreto).
  - Esquema do agente: deduz pelo enunciado; se omisso, infere um agente simples.
  - Algoritmos por defeito (se o LAB\_BRIEF não fixar outros):
      - **Q-Learning (tabular)**, por ser o mais didático para iniciantes.
  - Métricas por defeito:
      - **Recompensa Média Móvel** (para treino), **Taxa de Sucesso** e **Recompensa Média** (para avaliação).
  - Visualizações por defeito:
      - **Curva de Aprendizagem (Recompensa Média Móvel vs. Episódios)**.
      - **Heatmap da Função Valor (V(s))** (para ambientes em grelha como o FrozenLake).
  - Declara sempre num bloco “Assunções e Inferências” tudo o que assumiste.

ESTRUTURA OBRIGATÓRIA DO GUIÃO
Inclui **exatamente** as secções abaixo, com títulos, emojis e blocos de código dos prompts:

1)   Título do guião
2)   📚 Introdução ao Prompt Engineering  
           - 5 princípios (Sê Específico, Dá Contexto, Pede Exemplos, Itera, Estrutura a Tarefa)
3)   Bloco “Assunções e Inferências”  
           - Lista clara do que foi inferido ou assumido do LAB\_BRIEF (ambiente, tipo de tarefa, métricas, algoritmos, bibliotecas, ficheiros a gerar).
4)   PROMPT 1 — Definição e Exploração do Ambiente  
           - Importar `gymnasium` e criar o ambiente `{{ENV_NAME}}`.
           - Imprimir o **espaço de observação** (`observation_space`) e o **espaço de ações** (`action_space`).
           - Executar 10 passos aleatórios (`env.action_space.sample()`) e imprimir os resultados (`state, reward, terminated, truncated, info`).
           - Explicar o que são os espaços e o significado das recompensas (ex: *sparse reward*).
5)   PROMPT 2 — Definição do Agente (Q-Table e Hiperparâmetros)
           - Importar `numpy`.
           - Inicializar a **Q-Table** (tabela Q) com zeros (dimensões: estados x ações).
           - Definir os **hiperparâmetros** para Q-Learning:
              - `num_episodes` (ex: 10000), `max_steps_per_episode` (ex: 100).
              - `learning_rate` (alpha, $\alpha$), `discount_factor` (gamma, $\gamma$).
              - `epsilon` (taxa de exploração), `epsilon_decay_rate`, `min_epsilon`.
           - Explicar o papel de cada hiperparâmetro (Alpha, Gamma, Epsilon).
6)   PROMPT 3 — Treino do Agente (Loop Q-Learning)
           - Carregar (ou recriar) o ambiente, a Q-table e os hiperparâmetros.
           - Implementar o loop principal de treino (iteração por `num_episodes`).
           - Dentro de cada episódio, implementar a **política epsilon-greedy** para escolher a ação (explorar ou explorar).
           - Executar `env.step(action)` e obter `new_state, reward, terminated, truncated`.
           - Implementar a **atualização da Equação de Bellman** para a Q-Table:
              $Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha(R + \gamma \max_{a'} Q(s',a'))$
           - Implementar o **decaimento de epsilon** (linear ou exponencial).
           - Registar a recompensa total de cada episódio numa lista (`rewards_history`).
           - Guardar a Q-Table final (`q_table.npy`) e o histórico de recompensas (`rewards_history.pkl`).
7)   PROMPT 4 — Avaliação da Política (Métricas de Desempenho)
           - Carregar a `q_table.npy` treinada e recriar o ambiente.
           - Correr `n_eval_episodes` (ex: 100) com o agente em modo de avaliação.
           - **Sem exploração**: `epsilon = 0`. O agente deve escolher sempre a melhor ação (`np.argmax(q_table[state, :])`).
           - Calcular métricas: **Recompensa Média por Episódio** e **Taxa de Sucesso** (ex: % de episódios que atingiram o objetivo).
           - Tabela comparativa (Agente Treinado vs. Agente Aleatório).
           - Guardar CSV e Markdown.
8)   PROMPT 5 — Gráfico da Curva de Aprendizagem (Recompensas)
           - Carregar `rewards_history.pkl`.
           - Calcular uma **média móvel (moving average)** das recompensas (ex: janela de 100 episódios) para suavizar a curva.
           - Gerar um **gráfico de linha**: Média Móvel de Recompensa vs. Episódios.
           - Interpretação: O agente está a aprender? A recompensa converge para o máximo possível?
           - Guardar PNG e PDF.
9)   PROMPT 6 — Visualização da Política (Heatmap da Função Valor)
           - *Assumindo um ambiente em grelha como FrozenLake.*
           - Carregar a `q_table.npy`.
           - Calcular a **Função Valor (V(s))** para cada estado: $V(s) = \max_a Q(s,a)$.
           - Redimensionar $V(s)$ para o formato da grelha do ambiente (ex: `V.reshape(4, 4)` para FrozenLake 4x4).
           - Gerar um **heatmap (seaborn)** de $V(s)$ com anotações.
           - Interpretação: O estado-objetivo (Goal) tem o valor mais alto? Os estados de "buraco" (Hole) têm valor zero?
10) PROMPT 7 — Relatório Automático (Markdown)  
            - Geração do “RELATORIO\_FINAL.md” com: Introdução (Descrição do Ambiente), Agente (Q-Learning), Hiperparâmetros de Treino, **Curva de Aprendizagem (Gráfico)**, **Resultados da Avaliação (Tabela de Métricas)**, **Heatmap da Política (Função Valor)**, Conclusões e Recomendações.
            - Usar funções por secção; pathlib; pandas.
11) PROMPT 8 — Ficheiro Orquestrador  
            - `lab_orquestrador.py` (ou nome inferido do LAB\_BRIEF) que executa scripts na ordem;  
              verificação de artefactos, subprocess/argparse, mensagens de progresso, tempos, log “execucao.log”, seleção de etapas, tratamento de erros, opção continuar/abortar/tentar de novo, prints coloridos se possível.

FORMATO DE CADA PROMPT

  - Cabeçalho com emoji e título (ex.: “\#\# 🌍 PROMPT 1 — Definição e Exploração do Ambiente”).
  - Bloco “O que vais aprender” (3–5 bullets).
  - **Bloco de código** com o texto do prompt a enviar ao LLM, incluindo:
      - Nome do ficheiro a criar (ex.: `01_explora_ambiente.py`).
      - Requisitos técnicos concretos.
      - Bibliotecas a usar.
      - Exigir comentários extensos e prints.
  - Checklist “Após receber o código:” com passos claros (criar, colar, correr, verificar, etc.).

CONTRA-EXEMPLOS (NÃO FAZER)

  - Não inventes ambientes, algoritmos ou bibliotecas fora do LAB\_BRIEF sem declarar assunções.
  - Não alteres a ordem lógica Exploração Env → Definição Agente → Treino → Avaliação → Visualizações → Relatório → Orquestração.
  - Não omitas a guarda de artefactos (.npy, .pkl, .csv, .png/.pdf, .md).
  - Não uses gerúndios.

SAÍDA (OUTPUT)
Produz APENAS o documento final do “Guião de Prompts”, já pronto a copiar, contendo:

  - Títulos e emojis,
  - As 8 secções de PROMPTS com blocos de código,
  - As checklists pós-prompt,
  - A secção “Assunções e Inferências” no topo.
  - Adapta automaticamente métricas e gráficos à tarefa de Aprendizagem por Reforço.