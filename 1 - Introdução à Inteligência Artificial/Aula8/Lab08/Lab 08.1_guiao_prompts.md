# Guião de Prompts para Lab 08.1 — Aprendizagem por Reforço: Otimizar uma Política de Voo

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** identifica o ficheiro, entradas, algoritmos, métricas e artefactos.
2. **Dá contexto:** explica o cenário de rotas e navegação antes de pedir código.
3. **Pede exemplos:** solicita prints, tabelas e interpretações calculadas a partir dos resultados.
4. **Itera:** executa cada etapa e corrige erros antes de avançar.
5. **Estrutura a tarefa:** separa ambiente, bandit, treino, avaliação, visualização, relatório e orquestração.

## Assunções e Inferências

- **Pasta de trabalho:** `lab08.1_sol`, dentro de `Aula8/Lab08`.
- **Dados:** não existe CSV de entrada; os ambientes simulados geram recompensas durante a interação.
- **Ambientes:** um *Multi-Armed Bandit* com cinco rotas e um `GridWorld` discreto de 10×10.
- **GridWorld:** ações `cima`, `baixo`, `esquerda` e `direita`; recompensa `+10` no destino, `-10` num obstáculo e `-0.1` por passo normal.
- **Algoritmos Bandit:** Epsilon-Greedy, UCB e Thompson Sampling. O Thompson Sampling pode usar uma aproximação normal, pois as recompensas são contínuas.
- **Algoritmo de controlo:** Q-Learning tabular com política epsilon-greedy e decaimento de `epsilon`.
- **Métricas:** recompensa média acumulada, *regret* acumulado, recompensa por episódio, média móvel, taxa de sucesso e passos por episódio.
- **Reprodutibilidade:** usar sementes explícitas, por exemplo `42`, sempre que exista aleatoriedade.
- **Bibliotecas:** `pathlib`, `json`, `pickle`, `numpy`, `pandas`, `matplotlib` e `seaborn`. `gymnasium` é opcional: o `GridWorld` próprio torna o laboratório executável sem essa dependência.
- **Artefactos:** guardar tudo em `artefactos/modelos`, `artefactos/tabelas`, `artefactos/imagens`, `artefactos/relatorios` e `artefactos/logs`.
- **Regra transversal:** os scripts numerados executam de cima para baixo; não devem definir funções, `main` nem o bloco `if __name__ == "__main__":`. O módulo `rl_ambientes.py` é a única exceção, pois precisa de classes e métodos para representar os ambientes.

## 🌍 PROMPT 1 — Definição e Exploração dos Ambientes

### O que vais aprender

- Como representar estados, ações, transições e recompensas.
- A diferença entre o estado único de um Bandit e os estados de uma grelha.
- Como o *reward shaping* orienta o comportamento do agente.

```text
Cria `rl_ambientes.py` e `01_explora_ambientes.py` para o Lab 08.1, na pasta `lab08.1_sol`.

Usa `pathlib`, `json`, `numpy` e `pandas`. Escreve comentários extensos e prints informativos em português europeu. Usa caminhos relativos construídos com `pathlib` e cria `artefactos/tabelas` quando necessário.

Em `rl_ambientes.py`, implementa somente as classes necessárias:
- `MultiArmedBandit`, com nomes de rotas, médias e desvios-padrão; cada `step(acao)` devolve uma recompensa contínua amostrada para a rota escolhida;
- `GridWorld` 10x10, com conversão posição/índice, `reset`, `step` e amostragem de ação. Define ações 0=cima, 1=baixo, 2=esquerda e 3=direita, início, destino e obstáculos fixos e reprodutíveis. A colisão num obstáculo termina com -10; o destino termina com +10; cada passo normal vale -0.1. Movimentos fora da grelha não podem provocar erro.
Inclui um método simples de descrição em cada ambiente. Não uses Gymnasium como requisito obrigatório.

Em `01_explora_ambientes.py`, não cries funções, `main` nem bloco condicional. Importa as classes e:
1. Cria um Bandit com cinco rotas de voo, recompensas médias diferentes e semente fixa. Imprime ações disponíveis, estado único, espaço de ações e a explicação de que o agente não conhece as médias reais.
2. Faz 10 escolhas aleatórias, mostra rota e recompensa e guarda `artefactos/tabelas/exploracao_bandit.csv`.
3. Cria o GridWorld, imprime número de estados, número de ações, mapeamento das ações, posição inicial, destino, obstáculos e função de recompensa.
4. Executa até 10 ações aleatórias e imprime estado, posição, ação, novo estado, recompensa, terminação e motivo. Se terminar, faz `reset` e informa-o.
5. Guarda `artefactos/tabelas/exploracao_gridworld.csv` e `artefactos/tabelas/descricao_ambientes.json`.
6. Explica nos prints a diferença entre exploração e explotação, o significado de recompensa esparsa e por que a penalização por passo favorece caminhos curtos.

No fim, lista os artefactos criados. Não inventes dados externos nem conclusões não suportadas pela configuração do ambiente.
```

### Após receber o código:

- Guarda ambos os ficheiros em `lab08.1_sol`.
- Executa `python 01_explora_ambientes.py` nessa pasta.
- Confirma as transições, os limites da grelha e os dois CSV.

## 🎰 PROMPT 2 — Multi-Armed Bandit: Exploração vs. Explotação

### O que vais aprender

- Como Epsilon-Greedy, UCB e Thompson Sampling escolhem rotas.
- Como estimar o valor de cada ação por tentativa e erro.
- Como interpretar recompensa e *regret* acumulados.

```text
Cria `02_bandit_rotas.py` para o Lab 08.1. Não cries funções, `main` nem bloco condicional.

Usa `pathlib`, `pickle`, `numpy`, `pandas`, `matplotlib`, `seaborn` e `MultiArmedBandit` de `rl_ambientes`. Usa semente explícita e cria as pastas de artefactos. Escreve comentários abundantes e prints claros em português europeu.

Simula o mesmo Bandit de cinco rotas em pelo menos 30 repetições independentes e 2 500 passos por repetição. Compara:
1. Epsilon-Greedy, com `epsilon=0.1`;
2. UCB, que testa cada rota pelo menos uma vez e depois usa a estimativa mais um bónus de incerteza; usa e explica uma constante `c`;
3. Thompson Sampling aproximado para recompensas contínuas, com amostras normais centradas nas estimativas e incerteza maior para ações menos visitadas.

Para todos os algoritmos, atualiza as estimativas por média incremental, regista recompensa, ação, estimativas e contagens. Calcula o *regret* esperado por passo como `melhor_media_real - media_real_da_acao`; explica que esta referência só existe para avaliação da simulação, não para decisão do agente.

Agrega as repetições por algoritmo e passo. Guarda:
- `artefactos/tabelas/bandit_metricas_por_passo.csv`;
- `artefactos/tabelas/bandit_resumo.csv`, com recompensa final, regret final, percentagem de escolhas ótimas e ranking;
- `artefactos/modelos/bandit_config.pkl`;
- `artefactos/imagens/bandit_recompensa_media_acumulada.png` e `.pdf`;
- `artefactos/imagens/bandit_regret_acumulado.png` e `.pdf`.

Os gráficos devem mostrar uma linha por algoritmo, títulos e eixos em português. Imprime o ranking calculado, explica o compromisso exploração-explotação e evita afirmar que um algoritmo é universalmente superior: interpreta apenas as métricas desta experiência.
```

### Após receber o código:

- Executa `python 02_bandit_rotas.py`.
- Confirma que UCB explora todas as rotas no início.
- Compara as curvas e o *regret* final no resumo.

## 🧠 PROMPT 3 — Treino Q-Learning no GridWorld

### O que vais aprender

- Como criar uma Q-table de dimensão estados × ações.
- Como aplicar epsilon-greedy e a equação de Bellman.
- Como `alpha`, `gamma` e `epsilon` afetam a convergência.

```text
Cria `03_treino_qlearning.py` para o Lab 08.1. Não cries funções, `main` nem bloco condicional.

Usa `pathlib`, `json`, `pickle`, `numpy`, `pandas` e `GridWorld` de `rl_ambientes`. Cria as pastas de modelos e tabelas. Escreve comentários detalhados e prints em português europeu.

Cria um GridWorld com semente fixa e uma Q-table de zeros com forma `(n_states, n_actions)`. Define e imprime: `num_episodes=8000`, `max_steps_per_episode=200`, `learning_rate` (alpha), `discount_factor` (gamma), `epsilon` inicial, `min_epsilon` e taxa de decaimento exponencial. Explica cada valor e declara que a escolha serve este ambiente discreto, não uma regra universal.

Para cada episódio:
1. Faz reset ao ambiente e inicia recompensa, passos e sucesso a zero.
2. Escolhe ação aleatória com probabilidade `epsilon`; caso contrário escolhe uma ação de valor Q máximo, com desempate aleatório.
3. Executa a transição e atualiza a Q-table pela equação de Bellman: `Q(s,a) ← (1-alpha)Q(s,a) + alpha*(recompensa + gamma*max Q(s',a'))`. Em estado terminal, o alvo deve ser apenas a recompensa terminal.
4. Regista recompensa total, passos, sucesso, epsilon e médias móveis de 100 episódios para recompensa, sucesso e passos.
5. Aplica o decaimento, limitado por `min_epsilon`.

Imprime progresso a intervalos regulares, sem imprimir todos os episódios. Guarda `artefactos/modelos/q_table.npy`, `artefactos/modelos/qlearning_config.pkl`, `artefactos/tabelas/qlearning_historico_treino.csv` e uma cópia pickle do histórico. Explica, nos prints finais, os sinais de aprendizagem e porque uma curva isolada não prova convergência global.
```

### Após receber o código:

- Executa `python 03_treino_qlearning.py`.
- Confirma a forma da Q-table e a descida de `epsilon`.
- Verifica que o histórico inclui recompensa, sucesso, passos e médias móveis.

## ✅ PROMPT 4 — Avaliação da Política

### O que vais aprender

- Como separar treino e avaliação sem exploração.
- Como comparar uma política aprendida com uma política aleatória.
- Como calcular taxa de sucesso e eficiência do caminho.

```text
Cria `04_avaliacao_politica.py` para o Lab 08.1. Não cries funções, `main` nem bloco condicional.

Usa `pathlib`, `numpy`, `pandas` e `GridWorld`. Verifica a existência de `artefactos/modelos/q_table.npy`; se faltar, termina com um erro claro que indique a etapa anterior. Escreve comentários extensos e prints informativos em português europeu.

Carrega a Q-table e avalia, no mesmo GridWorld mas com uma semente de avaliação diferente, dois agentes durante 300 episódios e no máximo 200 passos:
- `agente_treinado`: epsilon igual a zero; escolhe ação de valor Q máximo e resolve empates aleatoriamente;
- `agente_aleatorio`: escolhe uma ação aleatória.

Para cada episódio regista recompensa total, sucesso, passos e motivo final. Para cada agente calcula episódios, recompensa média e desvio-padrão, taxa de sucesso, passos médios, recompensa total e percentis relevantes. Guarda `artefactos/tabelas/qlearning_avaliacao_metricas.csv`, a versão Markdown e `artefactos/tabelas/qlearning_avaliacao_episodios.csv`.

Imprime a comparação. Explica que `epsilon=0` serve para medir a política aprendida e que uma taxa de sucesso deve ser interpretada juntamente com passos e recompensa. Não alteres a Q-table nem repitas o treino.
```

### Após receber o código:

- Executa `python 04_avaliacao_politica.py`.
- Confirma que o agente treinado não explora.
- Consulta a tabela de métricas e o detalhe por episódio.

## 📈 PROMPT 5 — Gráficos de Aprendizagem

### O que vais aprender

- Como usar médias móveis para reduzir ruído.
- Como ler recompensa, sucesso, passos e exploração ao longo do treino.
- Por que deve existir cuidado ao concluir que o treino convergiu.

```text
Cria `05_graficos_aprendizagem.py` para o Lab 08.1. Não cries funções, `main` nem bloco condicional.

Usa `pathlib`, `pandas`, `matplotlib` e `seaborn`. Carrega `artefactos/tabelas/qlearning_historico_treino.csv` e valida as colunas necessárias. Se o ficheiro faltar, termina com mensagem clara. Cria `artefactos/imagens`, usa títulos e eixos em português europeu, guarda PNG a 300 dpi e PDF e fecha cada figura.

Gera:
1. Recompensa por episódio e recompensa média móvel de 100 episódios em `qlearning_curva_aprendizagem`;
2. Taxa de sucesso média móvel em `qlearning_taxa_sucesso`;
3. Passos médios móveis em `qlearning_passos_por_episodio`;
4. Evolução de epsilon em `qlearning_epsilon`.

Imprime valores inicial e final das métricas móveis disponíveis e uma interpretação prudente: melhoria sustentada de recompensa e sucesso, com menos passos, sugere aprendizagem; oscilações e estabilização devem ser verificadas antes de afirmar convergência. Não inventes interpretações nem modifiques o CSV de treino.
```

### Após receber o código:

- Executa `python 05_graficos_aprendizagem.py`.
- Confirma que cada gráfico existe em PNG e PDF.
- Verifica se o decaimento de epsilon coincide com a configuração do treino.

## 🧭 PROMPT 6 — Visualização da Política e Função Valor

### O que vais aprender

- Como extrair `V(s) = max_a Q(s,a)`.
- Como transformar os valores da Q-table num mapa interpretável.
- Como inspecionar setas, obstáculos, início e destino.

```text
Cria `06_visualizacao_politica.py` para o Lab 08.1. Não cries funções, `main` nem bloco condicional.

Usa `pathlib`, `numpy`, `pandas`, `matplotlib`, `seaborn` e `GridWorld`. Valida e carrega `artefactos/modelos/q_table.npy`; confirma que a forma coincide com o ambiente. Cria as pastas de tabelas e imagens. Comentários e textos devem estar em português europeu.

Calcula `V(s) = max_a Q(s,a)` e a ação de maior valor por estado. Redimensiona ambos para a grelha 10×10. Constrói anotações com `S` para início, `G` para destino, `X` para obstáculos e setas para os restantes estados.

Cria e guarda em PNG a 300 dpi e PDF:
- `artefactos/imagens/qlearning_funcao_valor_heatmap`, com valores V(s) anotados;
- `artefactos/imagens/qlearning_politica_setas`, com a política ótima, obstáculos, início e destino visíveis.

Guarda `artefactos/tabelas/qlearning_politica_grid.csv` com linha, coluna, estado, tipo, valor, melhor ação e símbolo. Imprime uma interpretação baseada nos valores e setas obtidos, incluindo a ressalva de que estados terminais podem ter valores particulares conforme a atualização de Bellman. Não inventes um caminho se as setas não o confirmarem.
```

### Após receber o código:

- Executa `python 06_visualizacao_politica.py`.
- Confirma que os obstáculos e o destino aparecem corretamente.
- Segue visualmente as setas desde `S` e verifica se evitam `X`.

## 📝 PROMPT 7 — Relatório Automático

### O que vais aprender

- Como reunir resultados sem voltar a treinar agentes.
- Como documentar métricas, decisões e limitações.
- Como criar um relatório reproduzível a partir de artefactos.

```text
Cria `07_relatorio_automatico.py` para o Lab 08.1. Não cries funções, `main` nem bloco condicional.

Usa `pathlib`, `pandas` e bibliotecas padrão. Lê os CSV existentes e verifica cada artefacto antes de o referir. Se algo for opcional e faltar, escreve “não disponível”; não inventes resultados e não executes treino, avaliação ou gráficos.

Gera `RELATORIO_FINAL.md` na raiz de `lab08.1_sol` e uma cópia em `artefactos/relatorios/RELATORIO_FINAL.md`. Inclui:
1. objetivo e cenários do laboratório;
2. descrição do Bandit e do GridWorld, estados, ações e recompensas;
3. comparação Epsilon-Greedy, UCB e Thompson Sampling, com tabela e gráficos de recompensa e regret;
4. Q-Learning, Q-table, política epsilon-greedy, alpha, gamma, equação de Bellman e decaimento de epsilon;
5. curva de aprendizagem, sucesso, passos e avaliação contra agente aleatório;
6. heatmap V(s) e mapa de setas;
7. discussão de exploração-explotação, reward shaping, estabilidade e convergência;
8. limitações, assunções e extensões, incluindo DQN apenas como extensão para espaços grandes.

Usa caminhos relativos corretos para imagens a partir de `artefactos/relatorios`. Inclui data e hora, tabela de artefactos com estado de existência e tabelas Markdown apenas quando os dados existirem. Imprime os caminhos dos relatórios criados.
```

### Após receber o código:

- Executa `python 07_relatorio_automatico.py`.
- Abre o relatório em Markdown e verifica as imagens e tabelas.
- Confirma que resultados em falta aparecem como “não disponível”.

## 🚀 PROMPT 8 — Orquestrador do Laboratório

### O que vais aprender

- Como executar um pipeline de RL pela ordem correta.
- Como registar duração, saída e falhas de cada etapa.
- Como repetir apenas as etapas necessárias.

```text
Cria `lab_orquestrador.py` para o Lab 08.1. Não cries funções, `main` nem bloco condicional.

Usa `pathlib`, `argparse`, `subprocess`, `sys`, `time`, `datetime`, `pandas` e bibliotecas padrão. Escreve mensagens de progresso em português europeu. Cria `artefactos/logs` e executa, usando `sys.executable`, as etapas nesta ordem:
1. `01_explora_ambientes.py`;
2. `02_bandit_rotas.py`;
3. `03_treino_qlearning.py`;
4. `04_avaliacao_politica.py`;
5. `05_graficos_aprendizagem.py`;
6. `06_visualizacao_politica.py`;
7. `07_relatorio_automatico.py`.

Antes de cada etapa, confirma que o script existe. Aceita `--etapas` com números de etapas, `--continuar-se-erro` e `--listar-etapas`. Captura stdout e stderr, mostra-os no terminal, acrescenta-os a `artefactos/logs/execucao.log` e mede a duração. Regista para cada etapa nome, estado, código de saída, duração e mensagem em `artefactos/logs/resumo_execucao.csv` e `.md`.

Se uma etapa falhar, termina por defeito com código diferente de zero; com `--continuar-se-erro`, regista a falha e avança. No final, confirma a presença dos artefactos principais e apresenta um resumo. Não apagues artefactos existentes, não instales dependências e não ocultes erros.
```

### Após receber o código:

- Executa `python lab_orquestrador.py --listar-etapas`.
- Corre o pipeline completo com `python lab_orquestrador.py`.
- Consulta `artefactos/logs/execucao.log`, o resumo e o relatório final.
