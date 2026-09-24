# Lab 08.2 - Algoritmos Genéticos — Otimização de Parâmetros e Rotas

**Tema:** Otimização Heurística com Algoritmos Genéticos

## 1. Objetivo

Neste laboratório, vais utilizar **Algoritmos Genéticos (AGs)** para resolver problemas de otimização complexos. Ao contrário dos métodos tradicionais que exploram um único ponto de cada vez, os AGs trabalham com uma "população" de soluções candidatas, evoluindo-as ao longo de gerações para encontrar ótimos globais.

Vamos focar-nos em dois cenários:
1.  **Otimização de Hiperparâmetros:** Encontrar a melhor combinação de hiperparâmetros para um modelo de Machine Learning.
2.  **Otimização Multi-objetivo:** Encontrar as melhores rotas de voo que equilibram dois objetivos concorrentes: **minimizar o tempo de voo** e **minimizar o consumo de combustível**.

## 2. O Espaço de Decisões

Em vez de um dataset tradicional, o teu "input" é o **espaço de possíveis soluções**.

**Cenário 1: Otimização de Hiperparâmetros**
-   **Indivíduo (Cromossoma):** Uma lista de hiperparâmetros para um modelo (e.g., XGBoost), como `[n_estimators, max_depth, learning_rate]`.
-   **Função de Fitness:** O desempenho do modelo (e.g., F1-Score ou ROC-AUC) treinado com esses hiperparâmetros. O objetivo é **maximizar** o fitness.

**Cenário 2: Otimização de Rota Multi-objetivo**
-   **Indivíduo (Cromossoma):** Uma sequência de waypoints que define uma rota.
-   **Funções de Fitness (Múltiplas):**
    1.  `f1(rota) = tempo_total_voo` (minimizar)
    2.  `f2(rota) = consumo_total_combustivel` (minimizar)
-   O objetivo é encontrar um conjunto de rotas que representem o melhor compromisso entre estes dois objetivos.

## 3. Tarefas

### Parte A: Otimização de Hiperparâmetros com um AG Clássico

1.  **Implementação:**
    *   Define a representação do cromossoma (os hiperparâmetros).
    *   Cria a **função de fitness**, que recebe um indivíduo, treina um modelo de ML e retorna a sua pontuação.
    *   Implementa ou utiliza uma biblioteca (e.g., `DEAP`, `geneticalgorithm`) para executar o ciclo do AG:
        *   **Seleção:** (e.g., Roleta, Torneio)
        *   **Crossover:** (e.g., Um Ponto, Dois Pontos)
        *   **Mutação:** (e.g., Inverter um bit, adicionar ruído gaussiano)
2.  **Avaliação:**
    *   Executa o AG por um número de gerações.
    *   Plota a evolução do **fitness máximo e médio** da população ao longo das gerações.
    *   Qual foi a melhor combinação de hiperparâmetros encontrada? Compara o desempenho com uma busca em grelha (`GridSearch`) ou aleatória (`RandomSearch`).

### Parte B: Otimização Multi-objetivo com NSGA-II

1.  **Implementação:**
    *   Utiliza um algoritmo especializado em multi-objetivo, como o **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**.
    *   Define as duas funções de fitness (tempo e consumo).
2.  **Avaliação:**
    *   O resultado de uma otimização multi-objetivo não é uma única solução, mas um conjunto de soluções ótimas não-dominadas, conhecido como a **Frente de Pareto**.
    *   Plota a Frente de Pareto final num gráfico de dispersão, com o tempo de voo num eixo e o consumo de combustível no outro.
    *   Analisa o gráfico. Ele mostra visualmente o *trade-off*: para diminuir o tempo de voo, tens de aceitar um maior consumo, e vice-versa.

## 4. Desafios e Lições

-   **Convergência Prematura:** Discute o que acontece se a diversidade da população for perdida muito cedo. Como os operadores de mutação e seleção ajudam a evitar que o algoritmo fique preso num ótimo local?
-   **Diversidade da População:** A diversidade é a chave para a exploração do espaço de busca. Reflete sobre como o NSGA-II usa mecanismos como o *crowding distance* para manter a diversidade ao longo da Frente de Pareto.
-   **Custo Computacional:** A função de fitness (especialmente no Cenário A) pode ser muito cara de calcular. Discute como isso torna os AGs uma ferramenta poderosa, mas computacionalmente intensiva.

## Produto Final

Um relatório que documente:  
-   A implementação do AG para otimização de hiperparâmetros, incluindo o gráfico de convergência do fitness.  
-   A implementação do NSGA-II para a otimização de rotas.  
-   O gráfico da **Frente de Pareto** resultante, com uma análise clara do que o *trade-off* significa no contexto do problema.  
-   Uma conclusão sobre a aplicabilidade dos AGs para problemas de otimização complexos e multi-objetivo.  