# Guião de Prompts para Lab 07.2 — Redução de Dimensionalidade: Visualizar e Simplificar Dados Complexos

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** indica o ficheiro a criar, o dataset, as colunas esperadas, os algoritmos, as métricas e os artefactos finais.
2. **Dá contexto:** explica que o objetivo é reduzir dados de sensores de voo com muitas dimensões, sem perder a estrutura essencial.
3. **Pede exemplos:** solicita interpretações dos componentes principais, dos loadings, dos gráficos 2D e do erro de reconstrução.
4. **Itera:** executa e valida cada etapa antes de avançar, para evitar que erros de escala ou de dados contaminem o pipeline.
5. **Estrutura a tarefa:** separa EDA, pré-processamento, seleção de componentes, transformação PCA, visualização, métodos não-lineares, relatório e orquestração.

## Assunções e Inferências

- **Dataset de origem:** `lab07.2/sensores_voo_alta_dim.csv`, relativo à pasta `Lab07`.
- **Cópia de trabalho:** `lab07.2_sol/dados/sensores_voo_alta_dim.csv`. Antes da primeira execução, copia o dataset de origem para esta localização.
- **Pasta de trabalho:** `lab07.2_sol`. Os scripts ficam diretamente nesta pasta, de acordo com a estrutura já usada na solução.
- **Features conhecidas:** `sensor_01`, `sensor_02`, ..., `sensor_50`.
- **Variável de referência:** `tipo_manobra`. Esta coluna serve para colorir visualizações e para treinar LDA. Não deve entrar no PCA, Kernel PCA, t-SNE ou UMAP.
- **Tipo de tarefa:** redução de dimensionalidade para interpretação, visualização e compressão de dados de sensores de voo.
- **Algoritmos principais:** PCA, Kernel PCA com kernel `rbf`, t-SNE, UMAP e LDA.
- **Papel do PCA:** método não supervisionado e linear; procura eixos de maior variância, sem usar `tipo_manobra`.
- **Papel do Kernel PCA:** método não supervisionado e não-linear; procura estruturas curvas ou separações que o PCA linear pode não revelar.
- **Papel do t-SNE e UMAP:** métodos não-lineares focados em visualização 2D; não substituem o PCA para análise de variância explicada.
- **Papel do LDA:** método supervisionado; usa `tipo_manobra` para procurar eixos que maximizam a separação entre classes.
- **Escalonamento:** `StandardScaler` é obrigatório antes de PCA, Kernel PCA, t-SNE, UMAP e LDA, porque sensores em escalas diferentes distorcem variância, distância e separação.
- **Métricas principais:** variância explicada acumulada, número de componentes para 95% de variância e erro de reconstrução MSE do PCA.
- **Interpretação:** os loadings do PCA indicam o peso de cada sensor em cada componente principal.
- **Estrutura de artefactos:** dados transformados em `artefactos`, modelos em `modelos`, tabelas em `tabelas`, imagens PNG/PDF em `imagens`, logs em `logs` e relatório em `relatorios`.
- **Reprodutibilidade:** usa `random_state=42` sempre que o algoritmo aceite este parâmetro.
- **Regra transversal do código:** cada script deve executar de cima para baixo. Não deve definir funções novas, uma função `main` ou um bloco `if __name__ == "__main__":`.
- **Conflito identificado no meta-prompt:** o Prompt 7 exige “funções por secção”, enquanto a regra transversal proíbe funções novas. Neste guião, a exceção fica limitada ao Prompt 7: pode criar funções pequenas para as secções do relatório, mas nunca uma função `main` nem o bloco `if __name__ == "__main__":`. Nos restantes prompts mantém-se a execução sequencial sem funções novas.

## 📊 PROMPT 1 — Análise Exploratória (EDA)

### O que vais aprender

- Como validar um dataset de alta dimensão antes de aplicar redução de dimensionalidade.
- Como estudar distribuição, skewness, outliers e escala das features.
- Como usar correlações para detetar redundância entre sensores.
- Como criar uma visualização inicial colorida por `tipo_manobra`.

```text
Cria o ficheiro `01_analise_exploratoria.py` para o Lab 07.2.

Contexto:
- O diretório de trabalho é `lab07.2_sol`.
- O dataset está em `dados/sensores_voo_alta_dim.csv`.
- O dataset tem sensores de voo de alta dimensão: `sensor_01`, `sensor_02`, ..., `sensor_50`.
- A coluna `tipo_manobra`, se existir, é um rótulo opcional. Usa-a para colorir gráficos exploratórios, mas não a uses como feature de PCA.

Regras de estrutura:
- Escreve todo o código em português europeu, com comentários abundantes e prints informativos.
- Organiza o script por blocos sequenciais bem identificados.
- Não cries funções novas, nem uma função `main`, nem o bloco `if __name__ == "__main__":`.
- Usa `pathlib`, `pandas`, `numpy`, `matplotlib` e `seaborn`.
- Usa caminhos relativos construídos com `pathlib`.
- Cria as pastas `imagens` e `tabelas` com `Path.mkdir(parents=True, exist_ok=True)`.

O script deve:
1. Validar se `dados/sensores_voo_alta_dim.csv` existe. Se faltar, termina com erro claro e não inventes dados.
2. Carregar o dataset e imprimir dimensão, primeiras linhas, últimas linhas, nomes das colunas, tipos, memória usada, valores em falta, duplicados e valores únicos.
3. Detetar automaticamente as features numéricas dos sensores. Excluir `tipo_manobra` das features numéricas caso surja codificada por engano.
4. Confirmar a presença de `tipo_manobra`, se existir, e imprimir a sua distribuição.
5. Calcular estatísticas descritivas para as features numéricas: média, desvio-padrão, mínimo, quartis, máximo, mediana e skewness.
6. Calcular outliers pelo método IQR, com Q1, Q3, IQR, limites inferior e superior, quantidade e percentagem por sensor.
7. Detetar valores infinitos e valores em falta. Não corrijas nem removas valores sem registo explícito.
8. Criar:
   - grelha de histogramas das features numéricas;
   - boxplots das features numéricas;
   - heatmap de correlação;
   - pairplot de 6 sensores com maior variância, colorido por `tipo_manobra` se existir.
9. Explicar nos prints e numa nota Markdown que o PCA é sensível à escala porque procura direções de maior variância.
10. Explicar que sensores com maior ordem de grandeza podem dominar o PCA sem escalonamento.
11. Guardar:
   - `tabelas/01_resumo_estatistico_sensores.csv`;
   - `tabelas/01_resumo_estatistico_sensores.md`;
   - `tabelas/01_matriz_correlacao.csv`;
   - `tabelas/01_notas_eda.md`;
   - imagens em PNG e PDF dentro de `imagens`.
12. No fim, imprime uma lista dos ficheiros criados e uma síntese curta baseada nos dados.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.2_sol/01_analise_exploratoria.py`.
- Copia o CSV original para `lab07.2_sol/dados/sensores_voo_alta_dim.csv`.
- A partir de `lab07.2_sol`, executa `python 01_analise_exploratoria.py`.
- Confirma que `tipo_manobra` não foi tratada como feature de PCA.
- Verifica se as tabelas e imagens foram criadas.

## 🧹 PROMPT 2 — Pré-processamento

### O que vais aprender

- Como separar features de sensores e rótulo de manobra.
- Por que o escalonamento é obrigatório para PCA.
- Como guardar `X_scaled`, `y` e transformadores.
- Como preparar dados para reutilização nas próximas etapas.

```text
Cria o ficheiro `02_preprocessamento.py` para o Lab 07.2.

Usa `pathlib`, `json`, `joblib`, `pandas`, `numpy` e scikit-learn, sobretudo `StandardScaler`.

Regras de estrutura:
- Escreve comentários abundantes e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais: carregamento, validação, separação, limpeza, encoding se necessário, escalonamento e gravação.
- Usa caminhos relativos com `pathlib`.

Entrada:
- `dados/sensores_voo_alta_dim.csv`.
- Coluna de rótulo esperada: `tipo_manobra`.
- Features esperadas: colunas numéricas dos sensores.

O script deve:
1. Validar o ficheiro, as colunas, os tipos, valores em falta, infinitos e duplicados.
2. Separar `X` com as features numéricas dos sensores e `y` com `tipo_manobra`, caso exista.
3. Remover duplicados exatos apenas se existirem, com print do número de linhas removidas.
4. Tratar valores em falta numéricos por mediana, caso existam. Guardar o imputador em `modelos/imputer_mediana.pkl`.
5. Se existirem features categóricas adicionais:
   - aplicar ordinal encoding apenas quando a ordem estiver documentada;
   - aplicar one-hot encoding quando não houver ordem;
   - não inventar ordens;
   - guardar o encoder em `modelos/encoder_features.pkl`.
6. Ajustar `StandardScaler` em `X` e criar `X_scaled`.
7. Explicar nos comentários, prints e notas:
   - PCA depende da variância;
   - sem escalonamento, sensores com valores maiores dominam os componentes;
   - Kernel PCA, t-SNE e UMAP também são afetados por escalas quando usam distâncias;
   - LDA também beneficia de escala comum quando as features têm ordens de grandeza diferentes.
8. Validar que `X_scaled` não tem NaN nem infinitos.
9. Imprimir médias e desvios-padrão antes e após o escalonamento.
10. Guardar:
   - `artefactos/X_original_features.csv`;
   - `artefactos/X_scaled.csv`;
   - `artefactos/y.csv`, se existir `tipo_manobra`;
   - `artefactos/feature_names.csv`;
   - `modelos/scaler_standard.pkl`;
   - `artefactos/preprocessamento_metadata.json`;
   - `tabelas/02_resumo_X_scaled.csv`;
   - `tabelas/02_resumo_X_scaled.md`.
11. No fim, imprime todos os artefactos criados.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.2_sol/02_preprocessamento.py`.
- Executa `python 02_preprocessamento.py`.
- Confirma que `X_scaled.csv` contém apenas features e não inclui `tipo_manobra`.
- Verifica se `y.csv`, `feature_names.csv` e `scaler_standard.pkl` existem.
- Lê os prints sobre médias e desvios-padrão após o escalonamento.

## 📐 PROMPT 3 — PCA: Seleção de Componentes (Scree Plot)

### O que vais aprender

- Como calcular a variância explicada por cada componente.
- Como escolher o número de componentes para reter 95% da variância.
- Como interpretar o Scree Plot e a curva de variância acumulada.
- Como quantificar o erro de reconstrução para vários valores de `k`.

```text
Cria o ficheiro `03_pca_selecao_componentes.py` para o Lab 07.2.

Usa `pathlib`, `json`, `pandas`, `numpy`, `matplotlib`, `seaborn` e `sklearn.decomposition.PCA`.

Regras de estrutura:
- Escreve comentários abundantes e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais.
- Usa caminhos relativos com `pathlib`.

Entrada:
- `artefactos/X_scaled.csv`.
- `artefactos/feature_names.csv`.

O script deve:
1. Carregar `X_scaled.csv` e validar que só contém valores numéricos, finitos e sem NaN.
2. Aplicar `PCA(n_components=None, random_state=42)` para calcular todos os componentes.
3. Calcular:
   - variância explicada por componente;
   - percentagem de variância explicada;
   - variância explicada acumulada;
   - número mínimo de componentes para atingir 95% da variância.
4. Criar uma tabela com componente, variância explicada, percentagem e percentagem acumulada.
5. Calcular o erro de reconstrução MSE para vários valores de `k`, desde 1 até ao número total de componentes.
6. Criar:
   - Scree Plot com barras da variância explicada por componente;
   - gráfico da variância explicada acumulada, com linha horizontal nos 95%;
   - gráfico do erro de reconstrução por `k`.
7. Explicar nos prints:
   - como escolher `k` pelo limiar de 95%;
   - como identificar um possível cotovelo visual;
   - por que há sempre compromisso entre compressão e perda de informação.
8. Guardar:
   - `modelos/pca_full.pkl`;
   - `tabelas/03_pca_variancia_componentes.csv`;
   - `tabelas/03_pca_variancia_componentes.md`;
   - `artefactos/pca_selecao_componentes_metadata.json`;
   - gráficos PNG e PDF em `imagens`.
9. No fim, imprime o número de componentes necessário para 95% de variância.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.2_sol/03_pca_selecao_componentes.py`.
- Executa `python 03_pca_selecao_componentes.py`.
- Abre a tabela de variância e confirma o `k` para 95%.
- Observa o Scree Plot e a curva acumulada.
- Verifica se o erro de reconstrução diminui quando `k` aumenta.

## 🧭 PROMPT 4 — PCA: Transformação e Análise de Loadings

### O que vais aprender

- Como transformar dados escalonados para o espaço PCA.
- Como guardar o modelo PCA final e os dados reduzidos.
- Como interpretar loadings.
- Como ligar componentes principais a sensores originais.

```text
Cria o ficheiro `04_pca_transformacao_loadings.py` para o Lab 07.2.

Usa `pathlib`, `json`, `joblib`, `pandas`, `numpy`, `matplotlib`, `seaborn` e `sklearn.decomposition.PCA`.

Regras de estrutura:
- Escreve comentários abundantes e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco `if __name__ == "__main__":`.
- Organiza o código em blocos sequenciais.
- Usa caminhos relativos com `pathlib`.

Entrada:
- `artefactos/X_scaled.csv`;
- `artefactos/feature_names.csv`;
- `artefactos/pca_selecao_componentes_metadata.json`.

O script deve:
1. Carregar `X_scaled.csv` e os nomes das features.
2. Ler o número de componentes recomendado para 95% de variância a partir dos metadados da etapa 3.
3. Treinar um `PCA` final com esse número de componentes e `random_state=42`.
4. Transformar `X_scaled` em `X_pca`.
5. Criar nomes de colunas `PC1`, `PC2`, ..., `PCk`.
6. Guardar:
   - `modelos/pca_model_95.pkl`;
   - `artefactos/X_pca.csv`;
   - `artefactos/pca_final_metadata.json`.
7. Extrair `pca.components_` e criar uma tabela de loadings com componentes nas linhas e sensores nas colunas.
8. Criar uma tabela com os sensores de maior peso absoluto em cada componente.
9. Gerar:
   - heatmap dos loadings;
   - gráfico dos maiores loadings absolutos de PC1;
   - gráfico dos maiores loadings absolutos de PC2, se existir.
10. Explicar nos prints:
   - loadings positivos e negativos;
   - sensores com maior peso absoluto;
   - por que PC1 e PC2 não são features originais, mas combinações lineares.
11. Guardar:
   - `tabelas/04_pca_loadings.csv`;
   - `tabelas/04_pca_loadings.md`;
   - `tabelas/04_top_loadings_por_componente.csv`;
   - `tabelas/04_top_loadings_por_componente.md`;
   - gráficos PNG e PDF em `imagens`.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.2_sol/04_pca_transformacao_loadings.py`.
- Executa `python 04_pca_transformacao_loadings.py`.
- Confirma a criação de `X_pca.csv` e `pca_model_95.pkl`.
- Consulta o heatmap de loadings.
- Regista quais sensores mais influenciam PC1 e PC2.

## 🔎 PROMPT 5 — Visualização PCA 2D e Erro de Reconstrução

### O que vais aprender

- Como visualizar PC1 e PC2 num gráfico 2D.
- Como usar `tipo_manobra` apenas como cor interpretativa.
- Como medir perda de informação por erro de reconstrução.
- Como identificar sensores com maior erro após compressão.

```text
Cria o ficheiro `05_visualizacao_pca_reconstrucao.py` para o Lab 07.2.

Usa `pathlib`, `json`, `joblib`, `pandas`, `numpy`, `matplotlib`, `seaborn` e métricas do scikit-learn.

Regras de estrutura:
- Escreve comentários abundantes e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais.
- Usa caminhos relativos com `pathlib`.

Entrada:
- `artefactos/X_scaled.csv`;
- `artefactos/X_pca.csv`;
- `artefactos/y.csv`, se existir;
- `modelos/pca_model_95.pkl`;
- `artefactos/feature_names.csv`.

O script deve:
1. Carregar os dados PCA e validar que existem pelo menos duas componentes.
2. Criar um gráfico de dispersão PC1 vs. PC2.
3. Colorir os pontos por `tipo_manobra` quando `y.csv` existir. Caso contrário, criar o gráfico sem cor por classe.
4. Usar títulos, eixos e legenda em português europeu.
5. Carregar o modelo PCA final e aplicar `inverse_transform` para reconstruir os dados escalonados.
6. Calcular:
   - MSE global de reconstrução;
   - RMSE global;
   - erro médio absoluto;
   - erro de reconstrução por feature.
7. Criar gráfico do erro de reconstrução por feature, ordenado do maior para o menor.
8. Explicar nos prints:
   - por que o erro de reconstrução mede informação perdida;
   - por que um `k` maior tende a reduzir o erro;
   - por que um gráfico 2D pode esconder variação que existe em componentes posteriores.
9. Guardar:
   - `tabelas/05_metricas_reconstrucao_pca.csv`;
   - `tabelas/05_metricas_reconstrucao_pca.md`;
   - `tabelas/05_erro_reconstrucao_por_feature.csv`;
   - `tabelas/05_erro_reconstrucao_por_feature.md`;
   - `artefactos/pca_reconstrucao_metadata.json`;
   - gráficos PNG e PDF em `imagens`.
10. No fim, imprime uma interpretação curta sobre separação visual e perda de informação.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.2_sol/05_visualizacao_pca_reconstrucao.py`.
- Executa `python 05_visualizacao_pca_reconstrucao.py`.
- Abre o gráfico PC1 vs. PC2.
- Confirma o MSE de reconstrução.
- Consulta quais sensores têm maior erro de reconstrução.

## 🌀 PROMPT 6 — Visualização Não-Linear (t-SNE e UMAP)

### O que vais aprender

- Como comparar PCA linear com métodos não-lineares.
- Como aplicar Kernel PCA, t-SNE e UMAP para visualização.
- Como distinguir métodos não supervisionados de LDA.
- Como interpretar separação visual entre tipos de manobra.

```text
Cria o ficheiro `06_visualizacao_nao_linear_lda.py` para o Lab 07.2.

Este prompt cumpre a etapa obrigatória de t-SNE e UMAP e inclui também Kernel PCA e LDA, porque ambos estão explicitamente pedidos no enunciado do Lab 07.2.

Usa `pathlib`, `json`, `joblib`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `KernelPCA`, `TSNE`, `LinearDiscriminantAnalysis` e `umap-learn`. A biblioteca `umap-learn` faz parte das dependências por defeito deste laboratório; se faltar, termina com uma mensagem clara que indique a instalação através de `requirements.txt`.

Regras de estrutura:
- Escreve comentários abundantes e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais.
- Usa caminhos relativos com `pathlib`.
- Usa `random_state=42` sempre que possível.

Entrada:
- `artefactos/X_scaled.csv`;
- `artefactos/y.csv`;
- `artefactos/X_pca.csv`, se já existir.

O script deve:
1. Carregar `X_scaled.csv` e validar os dados.
2. Carregar `y.csv`. Se não existir, executar apenas os métodos não supervisionados e criar gráficos sem cor por classe.
3. Se o dataset tiver mais de 10000 linhas, criar uma amostra aleatória de no máximo 5000 linhas para t-SNE e UMAP. Mantém a mesma amostra para `X` e `y`.
4. Aplicar `KernelPCA(n_components=2, kernel="rbf", gamma=None, random_state=42)` ou uma configuração equivalente compatível com a versão instalada do scikit-learn.
5. Aplicar `TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)`. Ajustar `perplexity` se houver poucas linhas.
6. Aplicar `UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)`. Se `umap-learn` não estiver instalado, terminar com um erro claro e instruções para instalar as dependências; não omitir o método silenciosamente.
7. Aplicar LDA apenas se `tipo_manobra` existir e tiver pelo menos duas classes. Usar `LinearDiscriminantAnalysis(n_components=min(2, n_classes - 1))`.
8. Explicar nos prints:
   - PCA e Kernel PCA não usam rótulos;
   - t-SNE e UMAP servem sobretudo para visualização;
   - LDA usa rótulos e por isso pode separar melhor as classes;
   - uma separação visual não prova, por si só, desempenho preditivo.
9. Criar gráficos 2D individuais para:
   - Kernel PCA;
   - t-SNE;
   - UMAP;
   - LDA, se aplicável.
10. Criar uma figura comparativa lado a lado com PCA, Kernel PCA, t-SNE, UMAP e LDA, consoante os dados disponíveis.
11. Guardar:
   - `artefactos/X_kernel_pca.csv`;
   - `artefactos/X_tsne.csv`;
   - `artefactos/X_umap.csv`;
   - `artefactos/X_lda.csv`, se aplicável;
   - `modelos/kernel_pca_rbf.pkl`;
   - `modelos/tsne_model.pkl`;
   - `modelos/umap_model.pkl`;
   - `modelos/lda_model.pkl`, se aplicável;
   - `modelos/label_encoder_tipo_manobra.pkl`, se aplicável;
   - `tabelas/06_resumo_metodos_visualizacao.csv`;
   - `tabelas/06_resumo_metodos_visualizacao.md`;
   - `artefactos/visualizacoes_nao_lineares_metadata.json`;
   - gráficos PNG e PDF em `imagens`.
12. No fim, imprime uma comparação curta entre PCA, Kernel PCA, t-SNE, UMAP e LDA.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.2_sol/06_visualizacao_nao_linear_lda.py`.
- Executa `python 06_visualizacao_nao_linear_lda.py`.
- Confirma que UMAP correu e que `X_umap.csv` e `umap_model.pkl` existem.
- Compara os gráficos 2D.
- Verifica se LDA ficou limitado pelo número de classes disponíveis.

## 📝 PROMPT 7 — Relatório Automático

### O que vais aprender

- Como transformar resultados técnicos num relatório legível.
- Como justificar o escalonamento e a escolha de componentes.
- Como comparar PCA, Kernel PCA, t-SNE, UMAP e LDA.
- Como apresentar limites e conclusões sem exagerar os resultados.

```text
Cria o ficheiro `07_relatorio_automatico.py` para o Lab 07.2.

Usa `pathlib`, `json`, `pandas` e leitura de ficheiros Markdown existentes. Não uses bibliotecas pesadas para gerar o relatório.

Regras de estrutura:
- Escreve comentários abundantes e prints informativos em português europeu.
- Esta é a exceção indicada no meta-prompt: cria funções pequenas, uma por secção do relatório, para organizar a geração de Markdown. Não cries uma função `main` nem o bloco `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais.
- Usa caminhos relativos com `pathlib`.
- Não inventes métricas nem conclusões. Lê os resultados dos ficheiros criados nas etapas anteriores.

Entrada:
- tabelas e metadados existentes em `tabelas` e `artefactos`;
- imagens existentes em `imagens`.

O script deve:
1. Criar `relatorios/RELATORIO_FINAL.md`.
2. Incluir título, data de geração e descrição do objetivo do Lab 07.2.
3. Incluir secção sobre dados:
   - dataset usado;
   - número de linhas e features, quando disponível;
   - papel de `tipo_manobra`.
4. Incluir secção de EDA:
   - distribuição das features;
   - outliers e skewness;
   - correlações;
   - motivo técnico para escalonamento.
5. Incluir secção de pré-processamento:
   - separação entre `X` e `y`;
   - exclusão de `tipo_manobra` dos métodos não supervisionados;
   - aplicação de `StandardScaler`.
6. Incluir secção PCA:
   - variância explicada;
   - número de componentes para 95%;
   - Scree Plot;
   - loadings;
   - interpretação dos primeiros componentes;
   - erro de reconstrução.
7. Incluir secção de visualizações:
   - PCA 2D;
   - Kernel PCA;
   - t-SNE;
   - UMAP, se existir;
   - LDA, se existir.
8. Incluir comparação explícita:
   - PCA linear vs. Kernel PCA/t-SNE/UMAP não-lineares;
   - PCA não supervisionado vs. LDA supervisionado;
   - visualização vs. compressão.
9. Incluir uma secção de limitações:
   - projeções 2D podem esconder informação;
   - t-SNE e UMAP dependem de hiperparâmetros;
   - LDA depende dos rótulos;
   - separação visual não equivale a validação preditiva.
10. Referenciar imagens com caminhos relativos em Markdown.
11. Guardar uma cópia opcional do relatório em `RELATORIO_FINAL.md` na raiz de `lab07.2_sol`.
12. No fim, imprime o caminho do relatório final e as secções incluídas.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.2_sol/07_relatorio_automatico.py`.
- Executa `python 07_relatorio_automatico.py`.
- Abre `relatorios/RELATORIO_FINAL.md`.
- Confirma que o relatório usa resultados reais.
- Verifica se as imagens estão referenciadas por caminhos relativos.

## 🧩 PROMPT 8 — Ficheiro Orquestrador

### O que vais aprender

- Como executar o laboratório completo com um único comando.
- Como selecionar etapas específicas.
- Como registar logs, tempos e erros.
- Como tornar o pipeline repetível.

```text
Cria o ficheiro `lab_orquestrador.py` para o Lab 07.2.

Usa `pathlib`, `argparse`, `subprocess`, `sys`, `time`, `datetime`, `csv` e `shutil` se for necessário copiar o dataset de origem.

Regras de estrutura:
- Escreve comentários abundantes e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais: argumentos, validação do dataset, seleção de etapas, execução, logs e resumo final.
- Usa caminhos relativos com `pathlib`.
- Executa os scripts com `sys.executable`.
- Guarda logs em `logs/execucao.log`.

Contexto:
- O diretório de trabalho é `lab07.2_sol`.
- Dataset esperado: `dados/sensores_voo_alta_dim.csv`.
- Dataset de origem possível: `../lab07.2/sensores_voo_alta_dim.csv`.
- Dependências por defeito: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib` e `umap-learn`.
- Scripts, pela ordem:
  1. `01_analise_exploratoria.py`
  2. `02_preprocessamento.py`
  3. `03_pca_selecao_componentes.py`
  4. `04_pca_transformacao_loadings.py`
  5. `05_visualizacao_pca_reconstrucao.py`
  6. `06_visualizacao_nao_linear_lda.py`
  7. `07_relatorio_automatico.py`

O orquestrador deve:
1. Criar as pastas `dados`, `artefactos`, `modelos`, `tabelas`, `imagens`, `relatorios` e `logs`.
2. Criar ou validar `requirements.txt` com as dependências por defeito, incluindo obrigatoriamente `umap-learn`.
3. Confirmar se o dataset existe em `dados/sensores_voo_alta_dim.csv`.
4. Se o dataset não existir, tentar copiar de `../lab07.2/sensores_voo_alta_dim.csv`. Se também faltar, apresentar erro claro. Não gerar dados sintéticos sem opção explícita.
5. Suportar:
   - `--listar` para listar etapas;
   - `--etapa 03` para executar só uma etapa;
   - `--desde 03` para executar desde uma etapa até ao fim;
   - `--ate 05` para executar até uma etapa;
   - `--continuar` para continuar após erro;
   - `--tentativas 2` para repetir uma etapa falhada.
6. Executar cada script com `subprocess.run`, `cwd` igual à pasta `lab07.2_sol` e captura de stdout/stderr.
7. Imprimir progresso, duração de cada etapa e estado final.
8. Guardar no log:
   - data e hora;
   - comando executado;
   - código de retorno;
   - duração;
   - stdout;
   - stderr.
9. Guardar também `logs/resumo_execucao.csv` com etapa, descrição, estado, duração e código de retorno.
10. Se uma etapa falhar:
   - registar a falha;
   - repetir conforme `--tentativas`;
   - abortar ou continuar conforme `--continuar`.
11. No fim, imprimir o caminho do relatório `relatorios/RELATORIO_FINAL.md`, se existir.
12. Terminar com código de saída 0 apenas se todas as etapas selecionadas terminarem com sucesso.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.2_sol/lab_orquestrador.py`.
- Executa `python lab_orquestrador.py --listar`.
- Executa `python lab_orquestrador.py`.
- Se precisares de testar uma parte, usa `python lab_orquestrador.py --desde 03 --ate 05`.
- Consulta `logs/execucao.log` e `logs/resumo_execucao.csv`.
- Confirma se `relatorios/RELATORIO_FINAL.md` foi criado no fim.
