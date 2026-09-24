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