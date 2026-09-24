# Guião de Prompts para Lab 07.1 — Clustering: Descoberta de Perfis Operacionais de Voo

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** indica o ficheiro a criar, os dados de entrada, os algoritmos, as métricas e os artefactos esperados.
2. **Dá contexto:** explica que a tarefa procura grupos latentes em telemetria de voo e que não existe variável alvo.
3. **Pede exemplos:** solicita interpretações concretas dos gráficos, das métricas e dos perfis descobertos.
4. **Itera:** executa e valida cada prompt antes de avançar para o seguinte, para evitar que um erro se propague pelo pipeline.
5. **Estrutura a tarefa:** separa EDA, pré-processamento, escolha de `k`, treino, visualização, interpretação, relatório e orquestração.

## Assunções e Inferências

- **Dataset de origem:** `lab07.1/voos_telemetria_completa.csv`, relativo à pasta `Lab07`.
- **Cópia de trabalho:** `lab07.1_sol/data/voos_telemetria_completa.csv`. Antes da primeira execução, copia o dataset de origem para esta localização.
- **Pasta de trabalho:** `lab07.1_sol`. Os scripts ficam na subpasta `scripts` e todos os resultados ficam na subpasta `outputs`.
- **Tipo de tarefa:** aprendizagem não supervisionada para descoberta de perfis operacionais de voo; não existe variável alvo.
- **Unidade de análise:** cada linha representa um voo e cada coluna representa uma métrica agregada desse voo.
- **Features conhecidas:** `duracao_voo_min`, `distancia_percorrida_km`, `altitude_maxima_m`, `velocidade_media_kmh`, `consumo_combustivel_litros` e `variacao_vertical_total_m`.
- **Tipos conhecidos:** as seis features são numéricas. Não se aplica encoding ao dataset fornecido; o guião inclui regras de segurança para eventuais colunas categóricas futuras.
- **Escalonamento principal:** `StandardScaler`, ajustado ao dataset completo porque não existe alvo nem divisão treino/teste nesta tarefa exploratória.
- **Outliers:** o pipeline identifica e documenta valores atípicos pelo método IQR, mas não os remove automaticamente. O DBSCAN serve também para estudar pontos de ruído.
- **Algoritmos principais:** K-Means, DBSCAN e Clustering Hierárquico Aglomerativo. O GMM permanece como extensão opcional do enunciado e não faz parte do pipeline mínimo.
- **Escolha de `k`:** combinação do Método Elbow com Coeficiente de Silhueta, Índice de Davies-Bouldin e Índice de Calinski-Harabasz. A decisão final deve incluir justificação técnica e operacional.
- **Métricas de comparação:** Silhueta, Davies-Bouldin e Calinski-Harabasz. Para o K-Means, usa-se também a Inércia, ou WCSS.
- **Visualização:** PCA com duas componentes para representar os clusters num plano. O PCA serve para visualização e não substitui os dados escalonados no treino.
- **Reprodutibilidade:** `random_state=42` sempre que o algoritmo aceite este parâmetro e `n_init=10` de forma explícita no K-Means.
- **Estrutura de artefactos:** modelos e transformadores em `outputs/models`, tabelas em `outputs/tables`, gráficos PNG/PDF em `outputs/figures`, notas e relatório em `outputs/reports`, e logs em `outputs/logs`.
- **Regra transversal do código:** cada script deve executar de cima para baixo. Não deve definir novas funções, uma função `main` ou um bloco `if __name__ == "__main__":`.

## 📊 PROMPT 1 — Análise Exploratória (EDA)

### O que vais aprender

- Como validar a estrutura e a qualidade de um dataset sem variável alvo.
- Como interpretar distribuições, assimetria e diferenças de escala.
- Como detetar outliers com o método IQR e avaliar o seu impacto no K-Means.
- Como usar pairplots e correlações para procurar potenciais grupos.

```text
Cria o ficheiro `scripts/01_analise_exploratoria.py` para o Lab 07.1.

Contexto:
- O diretório de trabalho é `lab07.1_sol`.
- O dataset está em `data/voos_telemetria_completa.csv`.
- Cada linha representa um voo.
- Não existe variável alvo: a tarefa é clustering.
- As colunas esperadas são:
  `duracao_voo_min`,
  `distancia_percorrida_km`,
  `altitude_maxima_m`,
  `velocidade_media_kmh`,
  `consumo_combustivel_litros`,
  `variacao_vertical_total_m`.

Regras de estrutura:
- Escreve todo o código em português europeu, com comentários extensos e prints informativos.
- Organiza o script por blocos sequenciais bem identificados.
- Não cries funções novas, nem uma função `main`, nem o bloco
  `if __name__ == "__main__":`.
- Usa `pathlib`, `pandas`, `numpy`, `matplotlib` e `seaborn`.
- Cria as pastas de saída necessárias com `Path.mkdir(parents=True, exist_ok=True)`.
- Usa caminhos construídos com `pathlib`, nunca caminhos absolutos.

O script deve:
1. Confirmar que o CSV existe. Se não existir, apresentar uma mensagem clara com o caminho esperado e terminar com erro, sem inventar dados.
2. Carregar o dataset e validar a presença exata das seis colunas esperadas. Assinalar colunas extra sem as apagar de forma silenciosa.
3. Imprimir dimensão, primeiras e últimas linhas, nomes das colunas, tipos, memória usada, valores em falta, duplicados e número de valores únicos.
4. Confirmar que as seis features são numéricas. Detetar valores infinitos e valores fisicamente impossíveis, como duração, distância, altitude, velocidade ou consumo negativos. Não corrigir esses valores sem justificação.
5. Calcular estatísticas descritivas com contagem, média, desvio-padrão, mínimo, quartis, máximo, mediana e skewness.
6. Criar uma tabela de outliers pelo método IQR, com Q1, Q3, IQR, limites inferior e superior, quantidade e percentagem de outliers por feature.
7. Explicar nos prints e em notas que:
   - features com escalas maiores podem dominar distâncias euclidianas;
   - assimetria forte pode deslocar centróides;
   - outliers podem atrair os centróides do K-Means e alterar os grupos;
   - transformação, winsorização, remoção justificada ou algoritmos robustos são opções, mas não devem ser aplicados sem análise.
8. Criar:
   - uma grelha de histogramas, com curva KDE quando fizer sentido;
   - uma grelha de boxplots;
   - um pairplot das seis features;
   - um heatmap da matriz de correlação;
   - histogramas e boxplots individuais para facilitar a leitura.
9. Colocar títulos, eixos e legendas em português europeu. Guardar cada figura em PNG a 300 dpi e em PDF. Fechar todas as figuras após a gravação.
10. Guardar:
   - `outputs/tables/eda_descritiva.csv`;
   - `outputs/tables/eda_descritiva.md`;
   - `outputs/tables/eda_outliers_iqr.csv`;
   - `outputs/tables/eda_outliers_iqr.md`;
   - `outputs/tables/eda_correlacoes.csv`;
   - gráficos em `outputs/figures/eda`;
   - `outputs/reports/NOTAS_EDA.md`.
11. Em `NOTAS_EDA.md`, incluir qualidade dos dados, diferenças de escala, features assimétricas, outliers, relações relevantes, potenciais agrupamentos visuais e justificação para o escalonamento.

No fim, imprime uma lista dos artefactos criados e um resumo curto das principais conclusões. Não inventes conclusões: calcula-as a partir dos dados.
```

### Após receber o código:

- Cria a pasta `lab07.1_sol/scripts` e guarda o ficheiro com o nome indicado.
- Copia o CSV original para `lab07.1_sol/data/voos_telemetria_completa.csv`.
- A partir de `lab07.1_sol`, executa `python scripts/01_analise_exploratoria.py`.
- Confirma os prints de validação e consulta `outputs/reports/NOTAS_EDA.md`.
- Verifica se as tabelas e todas as figuras existem e se os eixos estão legíveis.

## 🧹 PROMPT 2 — Pré-processamento

### O que vais aprender

- Por que o escalonamento é essencial em algoritmos baseados em distância.
- Como preservar uma versão original para interpretar os clusters.
- Como preparar features numéricas e lidar com eventuais categóricas.
- Como guardar dados processados e transformadores para reutilização.

```text
Cria o ficheiro `scripts/02_preprocessamento.py` para o Lab 07.1.

Usa `pathlib`, `json`, `pickle` ou `joblib`, `pandas`, `numpy` e componentes adequados do scikit-learn, sobretudo `StandardScaler`.

Regras de estrutura:
- Escreve comentários extensos e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco
  `if __name__ == "__main__":`.
- Organiza o código em blocos sequenciais: carregamento, validação, limpeza, encoding, escalonamento, validação final e gravação.
- Usa caminhos relativos com `pathlib`.

Entrada:
- `data/voos_telemetria_completa.csv`.
- Features esperadas:
  `duracao_voo_min`,
  `distancia_percorrida_km`,
  `altitude_maxima_m`,
  `velocidade_media_kmh`,
  `consumo_combustivel_litros`,
  `variacao_vertical_total_m`.
- Não existe variável alvo.

O script deve:
1. Validar o ficheiro, as colunas, os tipos, os valores em falta, os infinitos e os duplicados.
2. Manter apenas uma cópia dos duplicados exatos, caso existam, mas imprimir quantas linhas foram removidas. Não eliminar outliers.
3. Para valores em falta numéricos, aplicar imputação pela mediana e guardar o objeto de imputação. Explicar por que a mediana resiste melhor a outliers do que a média.
4. Confirmar que, no dataset atual, todas as features são numéricas e que não é necessário encoding.
5. Se surgirem colunas categóricas adicionais numa futura versão:
   - aplicar encoding ordinal apenas se existir uma ordem semântica documentada;
   - aplicar one-hot encoding quando não existir ordem;
   - não assumir nem inventar uma ordem;
   - guardar o encoder utilizado e os nomes finais das features.
6. Criar um DataFrame limpo nas escalas originais, com o índice das linhas preservado, para posterior interpretação dos clusters.
7. Ajustar `StandardScaler` ao conjunto completo de features e aplicar `fit_transform`. Como não existe alvo nem avaliação preditiva, não é necessário um split treino/teste.
8. Explicar de forma explícita, nos comentários, prints e notas:
   - K-Means usa distâncias euclidianas;
   - sem escala comum, uma feature com valores na ordem dos milhares pode dominar outra na ordem das dezenas;
   - o escalonamento centra cada feature em zero e atribui desvio-padrão unitário;
   - o escalonamento altera unidades, mas não remove outliers;
   - os dados originais continuam necessários para dar significado operacional aos clusters.
9. Validar que os dados escalonados não contêm NaN ou infinitos e imprimir médias e desvios-padrão antes e após o escalonamento.
10. Guardar:
   - `outputs/tables/dados_limpos.csv`;
   - `outputs/tables/dados_processados_sem_escala.csv`;
   - `outputs/tables/dados_escalonados.csv`;
   - `outputs/models/imputer_mediana.pkl`, se existir imputação;
   - `outputs/models/encoder.pkl`, se existir encoding;
   - `outputs/models/standard_scaler.pkl`;
   - `outputs/models/metadata_preprocessamento.json`;
   - `outputs/reports/NOTAS_PREPROCESSAMENTO.md`.
11. Guardar em `metadata_preprocessamento.json` as features originais, as features finais, o número de linhas, decisões de limpeza, estratégia de imputação, estratégia de encoding e parâmetros do scaler.

Não transformes distribuições nem removas outliers por defeito. Regista essas opções como experiências futuras. No fim, imprime todos os caminhos criados.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.1_sol/scripts/02_preprocessamento.py`.
- Executa `python scripts/02_preprocessamento.py` a partir de `lab07.1_sol`.
- Confirma que o número de linhas faz sentido após a remoção de duplicados.
- Abre os CSV escalonado e não escalonado e compara as ordens de grandeza.
- Confirma a criação do scaler, dos metadados e das notas de pré-processamento.

## 📐 PROMPT 3 — Método Elbow: Determinação de `k` para K-Means

### O que vais aprender

- Como a Inércia, ou WCSS, varia com o número de clusters.
- Como identificar o cotovelo sem tratar o gráfico como resposta absoluta.
- Como complementar o Método Elbow com três métricas internas.
- Como justificar uma escolha de `k` que faça sentido técnico e operacional.

```text
Cria o ficheiro `scripts/03_metodo_elbow.py` para determinar o número de clusters do K-Means.

Usa `pathlib`, `json`, `pandas`, `numpy`, `matplotlib`, `seaborn` e scikit-learn com `KMeans`, `silhouette_score`, `davies_bouldin_score` e `calinski_harabasz_score`.

Regras de estrutura:
- Escreve comentários extensos e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco
  `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais.
- Usa `random_state=42` e `n_init=10` em todos os modelos K-Means.

Entrada:
- `outputs/tables/dados_escalonados.csv`.
- Usa apenas as features processadas; uma coluna de índice acidental nunca pode entrar no modelo.

O script deve:
1. Carregar e validar os dados. Confirmar que são numéricos, finitos e que existem amostras suficientes.
2. Testar valores de `k` entre 2 e 10, sem ultrapassar `n_amostras - 1`.
3. Para cada `k`, ajustar um K-Means e calcular:
   - Inércia, também designada WCSS;
   - Coeficiente de Silhueta;
   - Índice de Davies-Bouldin;
   - Índice de Calinski-Harabasz.
4. Imprimir uma tabela formatada com todas as métricas.
5. Criar quatro gráficos:
   - Inércia em função de `k`, para o Método Elbow;
   - Silhueta em função de `k`;
   - Davies-Bouldin em função de `k`;
   - Calinski-Harabasz em função de `k`.
6. Criar também uma figura-resumo com os quatro gráficos.
7. Guardar todos os gráficos em PNG a 300 dpi e PDF em `outputs/figures/elbow`.
8. Tentar sugerir o cotovelo por um método numérico simples e transparente, como a maior distância à reta entre o primeiro e o último ponto da curva. Não uses bibliotecas adicionais só para detetar o cotovelo.
9. Comparar a sugestão do cotovelo com:
   - maior Silhueta;
   - menor Davies-Bouldin;
   - maior Calinski-Harabasz.
10. Escolher um `k` recomendado. Se as métricas discordarem, privilegiar Silhueta e uma solução simples, mas registar a discordância e não apresentar a escolha como verdade absoluta.
11. Explicar que:
   - a Inércia desce sempre à medida que `k` aumenta;
   - o cotovelo corresponde a uma redução do ganho marginal;
   - Silhueta perto de 1 indica grupos compactos e separados;
   - Davies-Bouldin mais baixo é melhor;
   - Calinski-Harabasz mais alto é melhor;
   - o objetivo operacional pode justificar outro `k`.
12. Guardar:
   - `outputs/tables/elbow_metricas_kmeans.csv`;
   - `outputs/tables/elbow_metricas_kmeans.md`;
   - `outputs/models/k_recomendado.json`, com o valor, critérios e métricas;
   - `outputs/reports/NOTAS_ELBOW.md`.

No fim, imprime o `k` recomendado, a justificação e os caminhos dos artefactos. Não peças ao utilizador para editar o script à mão.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.1_sol/scripts/03_metodo_elbow.py`.
- Executa `python scripts/03_metodo_elbow.py`.
- Compara visualmente o cotovelo com os máximos e mínimos das outras métricas.
- Confirma que `k_recomendado.json` contém uma escolha e a respetiva justificação.
- Regista qualquer decisão operacional diferente nas notas, sem apagar a recomendação automática.

## 🤖 PROMPT 4 — Treino de Modelos e Avaliação

### O que vais aprender

- Como comparar K-Means, DBSCAN e Clustering Hierárquico.
- Como interpretar parâmetros e métricas internas de clustering.
- Como um dendrograma revela a estrutura hierárquica dos dados.
- Como a ausência de escalonamento pode distorcer os resultados.
- Como os outliers afetam cada algoritmo.

```text
Cria o ficheiro `scripts/04_treino_avaliacao.py` para treinar e comparar modelos de clustering.

Usa `pathlib`, `json`, `pickle` ou `joblib`, `pandas`, `numpy`, `matplotlib`, scikit-learn e `scipy.cluster.hierarchy` apenas para criar o dendrograma.

Regras de estrutura:
- Escreve comentários extensos e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco
  `if __name__ == "__main__":`.
- Organiza o código em blocos sequenciais.
- Usa caminhos relativos com `pathlib`.

Entradas:
- `outputs/tables/dados_escalonados.csv`;
- `outputs/tables/dados_processados_sem_escala.csv`;
- `outputs/models/k_recomendado.json`.

O script deve:
1. Carregar os dados, remover apenas colunas de índice acidentais e validar a correspondência exata entre as linhas das versões escalada e original.
2. Ler automaticamente o `k` recomendado. Se o ficheiro não existir ou for inválido, terminar com uma mensagem que indique a execução prévia do Prompt 3.
3. Treinar:
   - K-Means nos dados escalonados, com `n_clusters=k`, `random_state=42` e `n_init=10`;
   - DBSCAN nos dados escalonados;
   - AgglomerativeClustering nos dados escalonados, com `n_clusters=k`.
4. Para o DBSCAN:
   - explicar que `eps` define o raio da vizinhança;
   - explicar que `min_samples` define a densidade mínima;
   - criar um gráfico das distâncias ao k-ésimo vizinho para apoiar a escolha de `eps`;
   - testar uma grelha pequena e documentada de `eps` e `min_samples`;
   - evitar uma pesquisa excessiva;
   - escolher uma combinação válida pela Silhueta, desde que produza pelo menos dois clusters;
   - contar clusters e pontos de ruído com label `-1`.
5. Criar um dendrograma com linkage de Ward sobre os dados escalonados. Se o dataset for muito grande, usar uma amostra reprodutível apenas no dendrograma e declarar essa opção. Guardar PNG a 300 dpi e PDF.
6. Calcular, para cada solução válida:
   - Coeficiente de Silhueta;
   - Índice de Davies-Bouldin;
   - Índice de Calinski-Harabasz;
   - número de clusters;
   - número e percentagem de pontos de ruído;
   - tempo de ajuste.
7. Para o DBSCAN, calcular as métricas sem os pontos `-1` e indicar esta regra na tabela. Se restar menos de dois clusters, registar métricas como indisponíveis sem provocar uma exceção.
8. Interpretar as métricas:
   - Silhueta mais alta e próxima de 1 é melhor;
   - Davies-Bouldin mais baixo e próximo de 0 é melhor;
   - Calinski-Harabasz mais alto é melhor;
   - nenhuma métrica isolada garante utilidade operacional.
9. Treinar ainda um K-Means diagnóstico nos dados sem escala, com o mesmo `k`. Comparar métricas e distribuição de labels com o K-Means escalonado. Marcar este resultado como experiência, não como candidato principal.
10. Explicar como outliers podem deslocar centróides no K-Means, formar ramos no hierárquico e surgir como ruído no DBSCAN.
11. Selecionar o melhor modelo principal pelo maior Coeficiente de Silhueta válido. Em empate, usar menor Davies-Bouldin e depois maior Calinski-Harabasz. Registar limitações de comparabilidade no caso do DBSCAN.
12. Guardar:
   - `outputs/tables/metricas_clustering.csv`;
   - `outputs/tables/metricas_clustering.md`;
   - `outputs/tables/labels_modelos.csv`;
   - `outputs/models/modelo_kmeans.pkl`;
   - `outputs/models/modelo_dbscan.pkl`;
   - `outputs/models/modelo_agglomerative.pkl`;
   - `outputs/models/parametros_dbscan.json`;
   - `outputs/models/melhor_modelo_clustering.txt`;
   - dendrograma e gráfico de k-distâncias em `outputs/figures/modelos`;
   - gráficos comparativos das três métricas em `outputs/figures/modelos`;
   - `outputs/reports/NOTAS_MODELOS.md`.

O ficheiro `labels_modelos.csv` deve manter o índice das amostras e ter colunas distintas para K-Means, DBSCAN, Agglomerative e K-Means sem escala. Guarda modelos com pickle/joblib e labels em CSV, pois nem todos os modelos atribuem clusters a novos dados da mesma forma.

No fim, imprime a tabela comparativa, o melhor modelo, a experiência sem escala e todos os artefactos criados.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.1_sol/scripts/04_treino_avaliacao.py`.
- Instala `scipy` se ainda não estiver disponível no ambiente.
- Executa `python scripts/04_treino_avaliacao.py`.
- Confirma que o DBSCAN não falha quando encontra um único cluster ou apenas ruído.
- Analisa o dendrograma, a tabela de métricas e a comparação do K-Means com e sem escala.

## 🗺️ PROMPT 5 — Visualização de Clusters com PCA

### O que vais aprender

- Como reduzir seis dimensões para duas dimensões de visualização.
- Como representar labels de clustering num plano PCA.
- Como projetar centróides do K-Means no mesmo espaço.
- Como distinguir separação visual de qualidade no espaço original.

```text
Cria o ficheiro `scripts/05_visualizacao_pca.py` para visualizar os clusters com PCA.

Usa `pathlib`, `json`, `pickle` ou `joblib`, `pandas`, `numpy`, `matplotlib`, `seaborn` e scikit-learn com `PCA`.

Regras de estrutura:
- Escreve comentários extensos e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco
  `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais.
- Não voltes a treinar os modelos.

Entradas:
- `outputs/tables/dados_escalonados.csv`;
- `outputs/tables/labels_modelos.csv`;
- `outputs/tables/metricas_clustering.csv`;
- modelos guardados em `outputs/models`;
- `outputs/models/melhor_modelo_clustering.txt`.

O script deve:
1. Validar que os dados e labels têm o mesmo número de linhas e a mesma ordem.
2. Ler o melhor modelo de forma automática. Confirmar a seleção pelo maior Coeficiente de Silhueta válido; em empate, usar menor Davies-Bouldin e maior Calinski-Harabasz.
3. Ajustar `PCA(n_components=2)` aos dados escalonados e criar as colunas `PC1` e `PC2`.
4. Imprimir e guardar a variância explicada de cada componente e a variância acumulada.
5. Explicar que o PCA reduz a dimensionalidade para uma representação 2D, mas pode perder informação. Uma eventual sobreposição no gráfico não prova que os clusters se sobrepõem no espaço original de seis dimensões.
6. Criar um scatter plot do melhor modelo, com:
   - uma cor por cluster;
   - legenda legível;
   - transparência e tamanho de ponto adequados;
   - eixos com a percentagem de variância explicada;
   - título em português europeu.
7. Criar também gráficos PCA individuais para K-Means e Agglomerative, para permitir a comparação pedida no laboratório.
8. Se o melhor modelo for K-Means, carregar os centróides, aplicar `pca.transform` aos centróides já escalonados e representá-los com um marcador grande e distinto.
9. No gráfico individual do K-Means, representar sempre os centróides projetados pelo PCA.
10. Criar uma figura lado a lado com K-Means e Agglomerative no mesmo espaço PCA e com limites de eixos iguais.
11. Guardar cada figura em PNG a 300 dpi e PDF em `outputs/figures/pca`.
12. Guardar:
   - `outputs/tables/componentes_pca_clusters.csv`;
   - `outputs/tables/pca_variancia_explicada.csv`;
   - `outputs/models/pca_2_componentes.pkl`;
   - `outputs/reports/NOTAS_PCA.md`.
13. Em `NOTAS_PCA.md`, indicar variância explicada, separação visual aparente, sobreposições, comparação K-Means/Agglomerative e limites da projeção.

Fecha todas as figuras. No fim, imprime a variância explicada, o modelo visualizado e os caminhos dos artefactos.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.1_sol/scripts/05_visualizacao_pca.py`.
- Executa `python scripts/05_visualizacao_pca.py`.
- Confirma que os labels mantêm a correspondência com as linhas dos dados.
- Verifica se os centróides surgem no gráfico do K-Means.
- Lê `NOTAS_PCA.md` antes de concluir que existe separação entre grupos.

## 🔎 PROMPT 6 — Visualização de DBSCAN e Perfil dos Clusters

### O que vais aprender

- Como o DBSCAN distingue clusters densos de pontos de ruído.
- Como interpretar clusters nas unidades originais das features.
- Como comparar médias, medianas e dimensão dos perfis.
- Como criar personas operacionais com base em evidência.

```text
Cria o ficheiro `scripts/06_dbscan_perfil_clusters.py` para visualizar o DBSCAN e interpretar os perfis do K-Means.

Usa `pathlib`, `json`, `pandas`, `numpy`, `matplotlib` e `seaborn`.

Regras de estrutura:
- Escreve comentários extensos e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco
  `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais.
- Não voltes a treinar os modelos nem a ajustar o PCA.

Entradas:
- `outputs/tables/dados_processados_sem_escala.csv`;
- `outputs/tables/dados_escalonados.csv`;
- `outputs/tables/labels_modelos.csv`;
- `outputs/tables/componentes_pca_clusters.csv`;
- `outputs/models/parametros_dbscan.json`.

O script deve:
1. Validar o número, a ordem e o índice das linhas em todas as tabelas.
2. Criar um gráfico PCA específico para DBSCAN:
   - uma cor por cluster válido;
   - pontos de ruído com label `-1` a cinzento ou preto;
   - marcador `X` para o ruído;
   - legenda que identifique claramente o ruído;
   - contagem e percentagem de ruído no subtítulo ou numa caixa de texto.
3. Guardar o gráfico DBSCAN em PNG a 300 dpi e PDF em `outputs/figures/perfil_clusters`.
4. Para o perfil operacional principal, juntar os labels do K-Means aos dados originais, não escalonados. Não uses valores z-score para dar nomes aos perfis.
5. Calcular por cluster:
   - quantidade de voos;
   - percentagem de voos;
   - média de cada feature;
   - mediana de cada feature;
   - desvio-padrão de cada feature.
6. Guardar a tabela completa com dados originais e cluster em `outputs/tables/dados_originais_com_clusters.csv`.
7. Guardar médias e medianas por cluster em CSV e Markdown.
8. Normalizar apenas a tabela de médias para criar:
   - um heatmap comparativo;
   - um gráfico de perfis por cluster.
   A normalização destes resumos serve apenas para o gráfico; os valores originais devem continuar disponíveis nas tabelas.
9. Criar uma persona para cada cluster com base em diferenças reais face à média ou mediana global. Usa nomes curtos e descritivos, por exemplo “Voos curtos e de baixa altitude”, apenas se os valores sustentarem essa descrição.
10. Para cada persona, indicar:
    - dimensão do cluster;
    - features acima e abaixo do valor global;
    - interpretação operacional;
    - cautelas e possíveis outliers.
11. Não inventar finalidades de missão que não constem dos dados. Não tratar o número do cluster como uma escala de qualidade.
12. Guardar:
    - `outputs/tables/perfil_clusters_medias.csv`;
    - `outputs/tables/perfil_clusters_medias.md`;
    - `outputs/tables/perfil_clusters_medianas.csv`;
    - `outputs/tables/personas_clusters.csv`;
    - `outputs/tables/personas_clusters.md`;
    - gráficos PNG/PDF em `outputs/figures/perfil_clusters`;
    - `outputs/reports/NOTAS_PERFIL_CLUSTERS.md`.
13. Em `NOTAS_PERFIL_CLUSTERS.md`, comparar K-Means com DBSCAN, discutir o ruído, interpretar as personas, descrever o efeito de outliers e indicar que os IDs dos clusters podem mudar entre execuções.

No fim, imprime as médias, as personas, a quantidade de ruído do DBSCAN e os caminhos criados.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.1_sol/scripts/06_dbscan_perfil_clusters.py`.
- Executa `python scripts/06_dbscan_perfil_clusters.py`.
- Confirma que o ruído do DBSCAN usa label `-1` e um marcador diferente.
- Valida os nomes das personas contra as médias e medianas em unidades originais.
- Revê os gráficos e as notas antes de aceitar a interpretação operacional.

## 📝 PROMPT 7 — Relatório Automático em Markdown

### O que vais aprender

- Como consolidar resultados dispersos num relatório reproduzível.
- Como incorporar tabelas e imagens sem inventar valores.
- Como relacionar EDA, escalonamento, métricas e perfis.
- Como comunicar limitações e recomendações de forma clara.

```text
Cria o ficheiro `scripts/07_relatorio_automatico.py` para gerar o relatório final do Lab 07.1.

Usa `pathlib`, `datetime`, `pandas`, `json` e bibliotecas padrão. Em cada secção, usa as funções adequadas de `pathlib` e `pandas`, mas não definas funções novas.

Regras de estrutura:
- Escreve comentários extensos e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco
  `if __name__ == "__main__":`.
- Organiza o código por blocos sequenciais, um por secção do relatório.
- Não voltes a treinar modelos, ajustar transformadores ou recalcular resultados que já existam.
- Lê os artefactos anteriores e apresenta “Não disponível” quando um artefacto opcional faltar.

O script deve gerar `outputs/reports/RELATORIO_FINAL.md` com:
1. Título, data e objetivo do laboratório.
2. Introdução à aprendizagem não supervisionada e ao problema de perfis operacionais de voo.
3. Descrição do dataset, número de voos, seis features e ausência de variável alvo.
4. Qualidade dos dados e EDA:
   - valores em falta e duplicados;
   - distribuição, skewness e outliers;
   - relações e correlações;
   - referência ao pairplot, histogramas, boxplots e heatmap.
5. Pré-processamento:
   - limpeza e imputação;
   - ausência ou presença de encoding;
   - StandardScaler;
   - explicação obrigatória da importância do escalonamento em algoritmos baseados em distância;
   - impacto dos outliers;
   - comparação K-Means com e sem escala.
6. Escolha de `k`:
   - Método Elbow e Inércia;
   - Silhueta, Davies-Bouldin e Calinski-Harabasz;
   - valor recomendado;
   - subjetividade e influência do objetivo operacional.
7. Modelos:
   - K-Means;
   - DBSCAN, com `eps`, `min_samples` e ruído;
   - AgglomerativeClustering e dendrograma.
8. Resultados:
   - tabela Markdown lida de `outputs/tables/metricas_clustering.csv`;
   - interpretação correta das três métricas;
   - melhor modelo e critérios de seleção;
   - limitações da comparação.
9. Visualização PCA:
   - variância explicada;
   - imagem do melhor modelo;
   - comparação K-Means/Agglomerative;
   - imagem DBSCAN com ruído;
   - aviso sobre perda de informação na projeção 2D.
10. Perfis dos clusters:
    - tabela de médias nas unidades originais;
    - tabela ou resumo das personas;
    - descrição detalhada de cada perfil;
    - dimensão absoluta e relativa de cada cluster.
11. Desafios e lições:
    - importância da normalização;
    - subjetividade de `k`;
    - efeito dos outliers;
    - estabilidade dos clusters.
12. Conclusões e recomendações:
    - interpretação operacional sem extrapolações;
    - validação com especialistas de domínio;
    - análise de estabilidade com várias sementes ou bootstrap;
    - teste de transformações para features assimétricas;
    - estudo de sensibilidade a outliers;
    - GMM como extensão opcional;
    - recolha de novas variáveis relevantes.
13. Inventário dos artefactos criados, com caminhos relativos.

Requisitos do Markdown:
- Usa títulos, listas e tabelas legíveis.
- Incorpora as imagens por caminhos relativos ao próprio relatório.
- Converte valores numéricos para uma apresentação adequada, sem alterar os CSV originais.
- Não inventes resultados, métricas, nomes de personas ou conclusões.
- Se um ficheiro obrigatório faltar, termina com uma mensagem clara que indique o prompt que deve ser executado.
- Cria também uma cópia do relatório na raiz de `lab07.1_sol` com o nome `RELATORIO_FINAL.md`.

No fim, imprime o caminho do relatório, o número de secções e a lista de artefactos em falta.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.1_sol/scripts/07_relatorio_automatico.py`.
- Executa `python scripts/07_relatorio_automatico.py`.
- Abre `outputs/reports/RELATORIO_FINAL.md` num visualizador de Markdown.
- Confirma que tabelas e imagens aparecem e que os caminhos relativos funcionam.
- Verifica cada conclusão contra os CSV e as notas produzidos pelas etapas anteriores.

## 🚀 PROMPT 8 — Ficheiro Orquestrador

### O que vais aprender

- Como executar um pipeline completo pela ordem correta.
- Como selecionar etapas e retomar uma execução interrompida.
- Como registar tempos, stdout, stderr e estados num log.
- Como tratar erros sem ocultar a etapa que falhou.

```text
Cria o ficheiro `lab_orquestrador.py` para executar o Lab 07.1 completo.

Usa `pathlib`, `subprocess`, `argparse`, `sys`, `time`, `logging`, `datetime`, `csv`, `shutil` e, apenas para o gerador de recurso, `numpy` e `pandas`.

Regras de estrutura:
- Escreve comentários extensos e prints informativos em português europeu.
- Não cries funções novas, nem uma função `main`, nem o bloco
  `if __name__ == "__main__":`.
- O programa deve executar de cima para baixo e usar `sys.executable` para chamar cada script.
- Usa códigos ANSI para mensagens coloridas apenas quando o terminal os suportar; o texto deve continuar legível sem cor.

Define, pela ordem, estas etapas:
1. `scripts/01_analise_exploratoria.py`
2. `scripts/02_preprocessamento.py`
3. `scripts/03_metodo_elbow.py`
4. `scripts/04_treino_avaliacao.py`
5. `scripts/05_visualizacao_pca.py`
6. `scripts/06_dbscan_perfil_clusters.py`
7. `scripts/07_relatorio_automatico.py`

O orquestrador deve:
1. Determinar a raiz do projeto a partir da localização de `lab_orquestrador.py`, sem depender do diretório atual da consola.
2. Criar `outputs/logs` e registar a execução em `outputs/logs/execucao.log`.
3. Validar a presença dos sete scripts e mostrar todos os ficheiros em falta antes de iniciar.
4. Validar `data/voos_telemetria_completa.csv`.
5. Se o dataset de trabalho faltar, procurar primeiro
   `../lab07.1/voos_telemetria_completa.csv` e copiá-lo para `data`.
6. Se nenhuma cópia original existir, terminar por defeito com instruções claras. Disponibilizar a opção explícita `--gerar-dados` para criar um dataset sintético de recurso:
   - usar exatamente as seis colunas conhecidas;
   - usar `numpy.random.default_rng(42)`;
   - criar vários perfis distinguíveis e alguns outliers;
   - impedir valores físicos negativos;
   - marcar no log e num ficheiro `data/DADOS_SINTETICOS.md` que os dados não são os dados originais e não devem sustentar conclusões reais.
7. Aceitar argumentos:
   - `--listar` para listar etapas;
   - `--etapa N` para executar uma etapa;
   - `--de N --ate M` para executar um intervalo;
   - `--continuar-em-erro`;
   - `--tentativas N`, com valor predefinido 1;
   - `--gerar-dados`;
   - `--sem-cor`.
8. Validar combinações de argumentos e apresentar ajuda útil para combinações inválidas.
9. Antes de cada etapa, imprimir número, nome, comando e hora de início.
10. Executar cada script com `subprocess.run`, `cwd` definido para a raiz do projeto, captura de stdout/stderr, texto e código de retorno.
11. Encaminhar stdout e stderr para a consola e para o log, sem ocultar mensagens de erro.
12. Medir duração, estado e número de tentativas de cada etapa.
13. Em caso de erro:
    - repetir até ao limite indicado;
    - após esgotar tentativas, abortar por defeito;
    - prosseguir apenas com `--continuar-em-erro`;
    - marcar etapas posteriores como potencialmente afetadas.
14. Guardar um resumo em:
    - `outputs/logs/resumo_execucao.csv`;
    - `outputs/logs/resumo_execucao.md`.
15. No fim, imprimir total de etapas concluídas, falhadas e ignoradas, tempo total, caminho do log e caminho do relatório final.
16. Terminar com código 0 apenas se todas as etapas solicitadas tiverem sucesso. Usar um código diferente de zero se alguma falhar.

Não apagues artefactos de execuções anteriores. Não instales dependências de forma automática. Não uses `shell=True`.
```

### Após receber o código:

- Guarda o ficheiro em `lab07.1_sol/lab_orquestrador.py`.
- Executa `python lab_orquestrador.py --listar` e confirma a ordem das sete etapas.
- Executa `python lab_orquestrador.py` para correr o pipeline completo.
- Usa `python lab_orquestrador.py --etapa 3` para testar a seleção de uma etapa.
- Consulta `outputs/logs/execucao.log`, o resumo da execução e `RELATORIO_FINAL.md`.
