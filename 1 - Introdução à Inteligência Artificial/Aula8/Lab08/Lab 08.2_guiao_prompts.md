# Guião de Prompts para Lab 08.2 — Algoritmos Genéticos: Otimização de Parâmetros e Rotas

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** indica ficheiros, entradas, representação do cromossoma, métricas e saídas.
2. **Dá contexto:** explica se o problema procura uma solução única ou uma frente de Pareto.
3. **Pede exemplos:** solicita tabelas, gráficos e interpretação dos resultados calculados.
4. **Itera:** executa e valida uma etapa antes da seguinte.
5. **Estrutura a tarefa:** separa dados, pré-processamento, AG clássico, avaliação, NSGA-II, relatório e orquestração.

## Assunções e Inferências

- **Pasta de trabalho:** `lab08.2_sol`, dentro de `Aula8/Lab08`.
- **Dados:** como o enunciado não fornece um dataset, o primeiro script cria um dataset sintético, determinístico e explicitamente identificado como pedagógico; não representa voos reais.
- **Tarefa supervisionada auxiliar:** classificação de risco/incidente de voo para demonstrar seleção de features e otimização de hiperparâmetros.
- **Modelo de fitness para seleção:** `LogisticRegression`, como definido pelo meta prompt; a métrica é acurácia média em validação cruzada. A otimização de hiperparâmetros mantém um modelo de árvore como cenário independente do enunciado.
- **AG clássico:** DEAP, indivíduos binários para seleção de features e cromossomas mistos/discretos para hiperparâmetros. A fitness usa validação cruzada apenas no treino.
- **Multi-objetivo:** NSGA-II, com pesos negativos para minimizar tempo e combustível. Cada rota é descrita por *waypoints* intermédios entre origem e destino.
- **Reprodutibilidade:** `random_state=42` e sementes explícitas em numpy e DEAP.
- **Artefactos:** `artefactos/eda`, `artefactos/preprocessamento`, `artefactos/ga_features`, `artefactos/avaliacao`, `artefactos/ga_hiperparametros`, `artefactos/nsga2_rotas`, `artefactos/relatorio` e `artefactos/logs`.
- **Regra de código:** cada script é sequencial e não cria funções, `main` ou bloco condicional, exceto as funções de fitness, mutação, validação de rota ou avaliação indispensáveis ao DEAP/NSGA-II. Todas as exceções devem estar documentadas.

## 🧪 PROMPT 1 — Dados Sintéticos e Análise Exploratória

### O que vais aprender

- Como criar dados pedagógicos de forma reprodutível.
- Como verificar qualidade, classes e relações entre variáveis.
- Como justificar a necessidade de otimização.

```text
Cria `01_gerar_dados_e_eda.py` para o Lab 08.2. Usa pathlib, numpy, pandas, matplotlib e seaborn; não cries funções, main nem bloco condicional. Escreve comentários e prints informativos em português europeu.

Cria, com semente 42, um dataset sintético de telemetria/risco de voo com pelo menos 800 linhas, uma coluna alvo binária `incidente` e features numéricas plausíveis, como duração, distância, altitude, velocidade, consumo, turbulência, vento, visibilidade, carga e idade da aeronave. Documenta que os dados são sintéticos e não inventes uma fonte real. Guarda `dados/dados_voo_ga.csv`.

Valida dimensão, tipos, nulos, duplicados, infinitos, estatísticas descritivas e distribuição da classe alvo. Cria histogramas, boxplots, heatmap de correlações e gráfico da distribuição do alvo; guarda PNG e PDF em `artefactos/eda`. Guarda tabelas de estatísticas e distribuição, mais `artefactos/eda/notas_eda.md`, que explique dimensionalidade, escalas, desequilíbrio, outliers e limitações. Não elimines linhas sem justificar.
```

### Após receber o código:

- Executa `python 01_gerar_dados_e_eda.py`.
- Confirma que o CSV é identificado como sintético e que todas as figuras existem.

## 🧹 PROMPT 2 — Pré-processamento e Divisão de Dados

### O que vais aprender

- Como evitar *data leakage*.
- Como preparar dados para fitness e teste final.
- Por que o escalonamento é ajustado apenas no treino.

```text
Cria `02_preprocessamento.py`. Usa pathlib, json, pickle/joblib, numpy, pandas e scikit-learn. Não cries funções, main nem bloco condicional.

Carrega `dados/dados_voo_ga.csv`, valida a coluna `incidente`, separa X e y e faz `train_test_split` estratificado, com `test_size=0.25` e `random_state=42`. Valida nulos, infinitos e tipos. Para features numéricas aplica imputação pela mediana, ajustada exclusivamente no treino, seguida de `StandardScaler`, também ajustado apenas no treino. Se existirem categorias futuras, define regras para ordinal apenas com ordem documentada e one-hot sem ordem.

Guarda X e y de treino/teste, escalados e com nomes de features preservados, em `artefactos/preprocessamento`; guarda scaler, imputador, nomes finais e `resumo_preprocessamento.json/.md`. Imprime a distribuição de classes antes e depois do split e explica a prevenção de leakage.
```

### Após receber o código:

- Verifica que treino e teste não partilham linhas.
- Confirma que o teste não entrou no `fit` do imputador nem do scaler.

## 🧬 PROMPT 3 — AG para Seleção de Features

### O que vais aprender

- Como uma lista binária representa um subconjunto.
- Como a fitness equilibra qualidade e complexidade.
- Como a validação cruzada protege a avaliação da fitness.

```text
Cria `03_ga_selecao_features.py`. Usa pathlib, json, pickle, numpy, pandas, DEAP e scikit-learn. Apenas a função `avaliar_individuo(individual)` pode ser definida; não cries outras funções, main nem bloco condicional.

Carrega os dados de treino escalados e os nomes das features. Cada indivíduo é uma lista de 0/1 com o mesmo comprimento das features. Configura DEAP com `FitnessMax`, indivíduo, seleção por torneio, crossover de dois pontos, mutação `flipBit`, população 50, crossover 0.7, mutação 0.2, elitismo via Hall of Fame e 40 gerações.

Em `avaliar_individuo`, devolve `(0,)` se nenhuma feature for selecionada; caso contrário calcula acurácia média com `cross_val_score`, `cv=3` e `LogisticRegression(max_iter=1000, random_state=42)` nos dados selecionados. Retorna a acurácia média num tuplo. Regista o número de features como estatística e discute uma penalização por complexidade, sem alterar a fitness base exigida pelo meta prompt.

Regista por geração fitness média, máxima, mínima, desvio-padrão e tamanho médio/mínimo do subconjunto. Guarda `hof.pkl`, `logbook.pkl`, população final, `best_features_preliminar.json`, CSV/Markdown de evolução e notas em `artefactos/ga_features`. Imprime o melhor indivíduo, a fitness e o número de features; não avalies no conjunto de teste nesta etapa.
```

### Após receber o código:

- Confirma que indivíduos vazios não provocam erro.
- Verifica que a fitness usa apenas treino e validação cruzada.

## 📈 PROMPT 4 — Convergência da Seleção de Features

### O que vais aprender

- Como ler fitness média e máxima por geração.
- Como detetar estagnação ou convergência prematura.
- Como relacionar qualidade com número de features.

```text
Cria `04_grafico_convergencia_ga.py`. Usa pathlib, pickle, pandas, matplotlib e seaborn; não cries funções, main nem bloco condicional.

Carrega o logbook do AG e extrai geração, fitness média, máxima e mínima, desvio-padrão e tamanho médio do subconjunto. Valida que existem dados antes de desenhar. Cria um gráfico de convergência da fitness e outro do tamanho do subconjunto, ambos em PNG a 300 dpi e PDF. Guarda também uma tabela limpa de evolução e `analise_convergencia.md` em `artefactos/ga_features`.

Interpreta apenas valores calculados: melhoria, estabilidade, diferença entre máximo e média e redução/variação de complexidade. Explica que um plateau não garante o ótimo global e que diversidade, mutação e população podem afetar convergência prematura.
```

### Após receber o código:

- Verifica se a fitness máxima melhora ou estabiliza.
- Compara a curva de fitness com o tamanho dos subconjuntos.

## ✅ PROMPT 5 — Avaliação do Melhor Subconjunto

### O que vais aprender

- Como usar o teste uma única vez após a procura.
- Como comparar o subconjunto com todas as features.
- Como interpretar F1 e matriz de confusão.

```text
Cria `05_avaliacao_melhor_subconjunto.py`. Usa pathlib, json, pickle/joblib, numpy, pandas, matplotlib, seaborn e métricas/modelos scikit-learn. Não cries funções, main nem bloco condicional.

Carrega Hall of Fame, nomes de features e os conjuntos de treino/teste. Extrai o melhor subconjunto e falha com mensagem clara se ficar vazio ou incompatível. Treina `LogisticRegression(max_iter=1000, random_state=42)` no treino selecionado e avalia no teste. Treina também o mesmo modelo com todas as features, para um baseline justo e com os mesmos parâmetros.

Calcula acurácia, precision, recall, F1, ROC-AUC quando possível e matriz de confusão. Guarda `best_features.json`, métricas CSV/Markdown, relatórios de classificação, modelo selecionado e matriz de confusão PNG/PDF em `artefactos/avaliacao`. Explica se houve ganho de acurácia, redução de features ou ambos, sem afirmar causalidade além da experiência.
```

### Após receber o código:

- Confirma que o teste não foi usado pelo AG.
- Compara acurácia, F1, recall e número de features com o baseline.

## ⚙️ PROMPT 6 — AG para Otimização de Hiperparâmetros

### O que vais aprender

- Como codificar hiperparâmetros num cromossoma.
- Como validar cada solução sem usar o teste.
- Como comparar AG com uma pesquisa aleatória simples.

```text
Cria `06_ga_hiperparametros.py`. Usa pathlib, json, pickle, numpy, pandas, DEAP e scikit-learn. Define apenas as funções indispensáveis de descodificação, reparação e fitness; não cries main nem outras funções auxiliares.

O cromossoma deve representar hiperparâmetros de Random Forest, por exemplo `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` e `max_features`; limita todos a intervalos válidos e inteiros quando necessário. A fitness é F1 média de validação cruzada estratificada no treino. Configura população, seleção, crossover, mutação, Hall of Fame, estatísticas e 40 gerações, com sementes explícitas.

Guarda melhor configuração JSON, logbook, modelo final treinado apenas no treino, métricas e gráficos de convergência em `artefactos/ga_hiperparametros`. Compara a melhor fitness de validação com uma RandomizedSearchCV de orçamento semelhante; não uses o teste nesta comparação. Explica custo computacional, risco de sobreajuste à validação e importância da diversidade.
```

### Após receber o código:

- Verifica limites válidos em cada cromossoma.
- Confirma que a comparação usa o mesmo orçamento aproximado.

## 🗺️ PROMPT 7 — NSGA-II e Frente de Pareto de Rotas

### O que vais aprender

- Como a não-dominação cria uma frente de Pareto.
- Como tempo e combustível entram em conflito.
- Como *crowding distance* protege diversidade.

```text
Cria `07_nsga2_rotas.py`. Usa pathlib, json, pickle, numpy, pandas, matplotlib, seaborn e DEAP. Define apenas as funções necessárias para reparar/avaliar uma rota; não cries main nem funções supérfluas.

Define origem e destino fixos e um cromossoma que codifique 3 a 5 waypoints intermédios num mapa bidimensional. Repara coordenadas para limites válidos. A função de fitness deve devolver `(tempo_total, combustivel_total)` e ambos devem ser minimizados: tempo depende da distância e condições simuladas; consumo aumenta com distância, mudanças de direção e condições adversas. Declara claramente que o modelo de custos é sintético e pedagógico.

Usa NSGA-II (`selNSGA2`), população de pelo menos 80, crossover/mutação adequados, 60 gerações e sementes. Guarda todos os indivíduos não dominados, waypoints, evolução de objetivos e `frente_pareto_rotas.csv`. Cria a frente de Pareto final e a evolução em PNG/PDF. Explica não-dominação, trade-off e crowding distance; não escolhe uma rota “melhor” sem um critério operacional explícito.
```

### Após receber o código:

- Confirma que ambos os objetivos são minimizados.
- Inspeciona se a frente apresenta alternativas tempo/combustível.

## 📝 PROMPT 8 — Relatório e Orquestrador

### O que vais aprender

- Como documentar resultados sem repetir cálculos.
- Como coordenar um pipeline reprodutível.
- Como guardar logs e estados de execução.

```text
Cria `08_relatorio_final.py` e `lab_orquestrador.py`. Os dois scripts não devem definir funções, main nem bloco condicional. Usa pathlib, pandas e bibliotecas padrão; o orquestrador usa ainda argparse, subprocess, sys, time e logging.

O relatório deve ler os artefactos existentes, usar “não disponível” quando faltarem e criar `RELATORIO_FINAL.md` na raiz e em `artefactos/relatorio`. Inclui dados sintéticos e limitações, seleção de features, convergência, avaliação no teste, AG de hiperparâmetros, NSGA-II, frente de Pareto, diversidade, custo computacional, conclusões e tabela de artefactos. Não volta a treinar nada nem inventa métricas.

O orquestrador executa as oito etapas numeradas na ordem, valida scripts, usa `sys.executable`, suporta `--etapas`, `--continuar-se-erro` e `--listar-etapas`, regista stdout/stderr, duração e código de saída em `artefactos/logs/execucao.log` e num resumo CSV/Markdown. Não instala dependências, não apaga artefactos e termina com código de erro se uma etapa falhar, exceto com a opção explícita para continuar.
```

### Após receber o código:

- Executa primeiro `python lab_orquestrador.py --listar-etapas`.
- Corre o pipeline e consulta o relatório, log e resumo de execução.
