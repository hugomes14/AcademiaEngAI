# Guião de Prompts para Lab 06.3 — Séries Temporais: Previsão de Procura e Consumo

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** indica ficheiros, colunas, horizontes e artefactos.
2. **Dá contexto:** explica o objetivo operacional e as limitações dos dados.
3. **Pede exemplos:** exige prints, tabelas e interpretações dos resultados.
4. **Itera:** executa cada etapa e corrige problemas antes da seguinte.
5. **Estrutura a tarefa:** separa EDA, preparação, treino, avaliação e relatório.

## Assunções e Inferências

- Existem dois ficheiros: `voltagem_bateria.csv` e `missoes_diarias.csv`.
- A primeira série usa `timestamp` e `voltagem`, com horizonte de 60 minutos.
- A segunda série usa `data` e `num_missoes`, com horizonte de 28 dias.
- A validação deve respeitar a ordem temporal; não se usa divisão aleatória.
- Métricas principais: MAE, MSE, RMSE e MAPE, com cuidado quando o valor real é zero.
- Modelos previstos: baseline de persistência, ARIMA/SARIMA, Holt-Winters, Prophet e, opcionalmente, LSTM.
- Prophet e LSTM devem ser tratados como opcionais quando não estiverem instalados.
- Todos os artefactos devem ficar em `artefactos/`.
- Nenhum script deve criar funções, `main` ou o bloco `if __name__ == "__main__":`.

## 📊 PROMPT 1 — Análise Exploratória das Séries Temporais

### O que vais aprender

- Como validar datas, frequência, lacunas e duplicados.
- Como identificar tendência, sazonalidade, anomalias e estacionariedade.
- Como usar decomposição, ACF, PACF e o teste ADF.

```text
Cria um script Python chamado `01_analise_exploratoria.py` para analisar `voltagem_bateria.csv` e `missoes_diarias.csv`.

Usa código sequencial no corpo do script. Não cries funções, `main` nem `if __name__ == "__main__":`.
Usa pathlib, pandas, numpy, matplotlib, seaborn e statsmodels.

Cria `artefactos/eda`, `artefactos/tabelas` e `artefactos/graficos`. Verifica a existência dos dois CSV e termina com uma mensagem clara se algum faltar.

Para cada ficheiro:
- confirma as colunas esperadas;
- converte a coluna temporal para datetime e ordena os dados;
- verifica frequência, duplicados, lacunas, valores em falta e valores impossíveis;
- imprime dimensão, tipos, primeiras linhas e estatísticas descritivas;
- cria um gráfico temporal da série;
- analisa tendência, sazonalidade e anomalias;
- executa ADF e interpreta o resultado sem afirmar causalidade;
- cria gráficos ACF e PACF;
- usa `seasonal_decompose` quando existirem dados suficientes e documenta o período escolhido.

Guarda todos os gráficos em PNG a 300 dpi e PDF, tabelas CSV e `artefactos/eda/resumo_eda.md`.
Escreve comentários pedagógicos, prints informativos em português europeu e fecha todas as figuras.
``` 

### Após receber o código:

- Guarda o ficheiro na pasta do laboratório.
- Confirma os dois CSV e executa o script.
- Verifica os gráficos, a frequência detetada e o relatório da EDA.

## 🧹 PROMPT 2 — Pré-processamento e Engenharia de Features

### O que vais aprender

- Por que o split temporal evita data leakage.
- Como criar lags, janelas móveis e variáveis de calendário.
- Como ajustar transformadores apenas no passado.

```text
Cria `02_preprocessamento.py`, sem funções, `main` ou bloco de execução condicional.
Usa pathlib, pickle ou joblib, pandas, numpy e scikit-learn.

Prepara as duas séries separadamente. Usa os 20% mais recentes como teste e os 80% anteriores como treino. Nunca embaralhes os dados.

Para a voltagem cria lags adequados à frequência, incluindo pelo menos 1, 5, 15 e 60 períodos, quando existirem dados suficientes. Para missões cria lags 1, 7 e 14 e features de calendário como dia da semana, mês e dia do ano. Cria médias móveis apenas com informação passada.

Remove NaN resultantes dos lags sem usar informação futura. Explica por que um split aleatório e uma janela móvel centrada provocariam data leakage.

Guarda X_train, X_test, y_train, y_test, índices temporais, nomes das features, metadados e os objetos de pré-processamento em `artefactos/dados` e `artefactos/modelos`. Não transforma o alvo sem documentar a transformação inversa.

Imprime as dimensões, os limites temporais de treino/teste e exemplos das matrizes. Cria `artefactos/preprocessamento/resumo_preprocessamento.md`.
``` 

### Após receber o código:

- Executa a EDA antes deste script.
- Confirma que todas as datas de teste são posteriores às de treino.
- Verifica que não existem NaN nas matrizes finais.

## 🤖 PROMPT 3 — Treino dos Modelos de Previsão

### O que vais aprender

- Como estabelecer um baseline de persistência.
- Como treinar modelos clássicos e modelos de regressão com lags.
- Como guardar previsões e intervalos sem contaminar o teste.

```text
Cria `03_treino_modelos.py`, apenas com código sequencial, sem funções nem `main`.
Usa pathlib, time, pickle/joblib, pandas, numpy, scikit-learn e statsmodels.

Carrega os dados processados. Para cada série treina:
- baseline de persistência;
- ARIMA ou SARIMA se a EDA justificar a sazonalidade;
- Holt-Winters/ExponentialSmoothing;
- Regressão Linear com lags e calendário;
- Ridge;
- Random Forest Regressor.

Para a voltagem gera previsões dos próximos 60 minutos. Para missões gera previsões dos próximos 28 dias. Ajusta cada modelo apenas com treino. Quando o modelo suportar, guarda limites inferior e superior dos intervalos de previsão.

Regista modelo, parâmetros, tempo, número de observações e horizonte. Guarda modelos, previsões de treino/teste e nomes das features em `artefactos/modelos` e `artefactos/previsoes`.
Não calcules métricas neste script. Imprime mensagens de progresso e avisa quando Prophet ou LSTM não estiverem instalados.
``` 

### Após receber o código:

- Confirma que o treino usa apenas o período histórico.
- Verifica se existem previsões para ambos os horizontes.
- Confirma que os ficheiros dos modelos foram criados.

## 📏 PROMPT 4 — Avaliação e Comparação

### O que vais aprender

- Como comparar previsões temporais com um baseline.
- Como interpretar MAE, MSE, RMSE e MAPE.
- Como evitar conclusões baseadas apenas numa métrica.

```text
Cria `04_avaliacao_modelos.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e sklearn.metrics.

Carrega as previsões de cada série e valida datas, colunas, valores em falta, ordem temporal e alinhamento entre real e previsto.

Calcula por modelo e por série MAE, MSE, RMSE e MAPE. Quando o valor real for zero, documenta e aplica uma versão segura do MAPE ou sMAPE. Inclui sempre o baseline de persistência.

Ordena por RMSE, identifica o melhor modelo e calcula a melhoria percentual face ao baseline. Guarda `artefactos/tabelas/metricas_modelos.csv`, uma versão Markdown com quatro casas decimais e `artefactos/avaliacao/resumo_avaliacao.md`.

Explica que o RMSE penaliza mais erros grandes, que o MAPE pode ser instável perto de zero e que uma validação temporal única limita a generalização.
``` 

### Após receber o código:

- Confirma que o baseline aparece na tabela.
- Verifica se o melhor modelo vence o baseline.
- Analisa separadamente voltagem e missões.

## 📈 PROMPT 5 — Gráfico Previsão vs. Real

### O que vais aprender

- Como visualizar previsões no eixo temporal correto.
- Como comparar histórico, valores reais e previsões.
- Como interpretar falhas em tendências e sazonalidades.

```text
Cria `05_previsto_vs_real.py`, sem funções nem `main`.
Usa pathlib, pandas, numpy, matplotlib e seaborn.

Seleciona automaticamente o melhor modelo de cada série pelo menor RMSE. Cria gráficos de linha que mostrem o histórico de treino, os valores reais do teste e as previsões no mesmo eixo temporal. Mostra, quando existirem, intervalos de previsão.

Usa títulos, legendas e unidades em português. Guarda PNG a 300 dpi e PDF em `artefactos/graficos`. Guarda também uma tabela com os maiores erros e uma interpretação em Markdown.

Imprime se o modelo acompanha a tendência, a sazonalidade, picos, quedas e alterações de variância. Não inventes previsões nem alteres os dados originais.
``` 

### Após receber o código:

- Confirma que o eixo temporal está ordenado.
- Verifica que o treino não é confundido com o teste.
- Observa os maiores erros e os intervalos de previsão.

## 📉 PROMPT 6 — Análise de Resíduos

### O que vais aprender

- Como calcular resíduos com a convenção `real - previsto`.
- Como detetar autocorrelação, tendência e heterocedasticidade.
- Por que um histograma não prova normalidade.

```text
Cria `06_analise_residuos.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy, matplotlib, seaborn e statsmodels.

Seleciona automaticamente o melhor modelo de cada série. Calcula `residuo = valor_real - valor_previsto`, resíduos absolutos, média, mediana, desvio-padrão, quartis, skewness, maior erro e correlação entre resíduos e previsões.

Cria histograma dos resíduos com linha em zero, resíduos ao longo do tempo, resíduos contra previsões e ACF dos resíduos. Se necessário usa um teste estatístico apenas como apoio e explica as suas limitações.

Interpreta centragem, caudas, autocorrelação, padrões sazonais, heterocedasticidade e erros maiores em determinados períodos. Guarda gráficos PNG/PDF a 300 dpi, `maiores_residuos.csv` e `artefactos/avaliacao/resumo_residuos.md`.
``` 

### Após receber o código:

- Confirma a convenção dos resíduos.
- Verifica se a autocorrelação residual diminuiu.
- Regista padrões que possam justificar mais lags ou outro modelo.

## 📝 PROMPT 7 — Relatório Automático em Markdown

### O que vais aprender

- Como reunir artefactos sem voltar a treinar modelos.
- Como documentar decisões e limitações.
- Como criar um relatório reproduzível.

```text
Cria `07_gerar_relatorio.py` sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e bibliotecas padrão.

Verifica artefactos opcionais e usa “não disponível” quando necessário. Gera `RELATORIO_FINAL.md` e uma cópia em `artefactos/relatorios/RELATORIO_FINAL.md` com:
1. Introdução;
2. Dados e qualidade;
3. EDA, decomposição, ACF e estacionariedade;
4. Pré-processamento e prevenção de data leakage;
5. Modelos e horizontes;
6. Métricas e comparação com baseline;
7. Gráficos de previsão;
8. Resíduos;
9. Discussão de ARIMA, Holt-Winters, Prophet e LSTM;
10. Conclusões e recomendações;
11. Assunções, limitações e referências.

Usa caminhos relativos corretos para as imagens, data/hora de geração e uma tabela de artefactos com existência confirmada. Não inventes resultados nem voltes a treinar modelos.
``` 

### Após receber o código:

- Confirma que os valores vêm dos CSV e Markdown existentes.
- Abre o relatório e verifica as ligações às imagens.
- Confirma que as limitações estão explícitas.

## 🚀 PROMPT 8 — Orquestrador do Laboratório

### O que vais aprender

- Como executar um pipeline temporal pela ordem correta.
- Como guardar logs, tempos e estados.
- Como tratar falhas sem esconder erros.

```text
Cria `lab_orquestrador.py` usando apenas código sequencial, sem funções, `main` ou bloco condicional.
Usa pathlib, subprocess, argparse, sys, time, logging e bibliotecas padrão.

Executa pela ordem: 01 EDA, 02 pré-processamento, 03 treino, 04 avaliação, 05 gráficos, 06 resíduos e 07 relatório.

Cria `artefactos/logs`, valida os dois datasets e os scripts antes de começar, usa `sys.executable`, captura stdout/stderr, regista duração e escreve `execucao.log`. Aceita execução de todas as etapas, de uma etapa, de um intervalo, `--continuar-em-erro`, `--tentar-novamente` e `--listar-etapas`.

Não gera dados fictícios por defeito. Se existir uma opção explícita para gerar dados pedagógicos, identifica-os claramente como sintéticos. Apresenta resumo final, código de saída coerente e guarda `resumo_execucao.csv` e `resumo_execucao.md`.
``` 

### Após receber o código:

- Executa primeiro `--listar-etapas`.
- Corre o pipeline completo.
- Verifica o log, o relatório e o estado final de cada etapa.
