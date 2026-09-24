# Guião de Prompts para Lab 06.1 — Regressão: Estimar a Duração de Voo

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** indica dataset, alvo, features e artefactos.
2. **Dá contexto:** explica o problema operacional e as unidades.
3. **Pede exemplos:** exige prints, tabelas e interpretações.
4. **Itera:** executa e valida cada etapa antes da seguinte.
5. **Estrutura a tarefa:** separa EDA, pré-processamento, treino e avaliação.

## Assunções e Inferências

- Dataset: `voos_telemetria.csv`.
- Alvo: `duracao_voo_min`.
- Features numéricas: `distancia_planeada`, `carga_util_kg` e `altitude_media_m`.
- Feature categórica: `condicao_meteo`, com ordem `Bom < Moderado < Adverso`.
- Problema: regressão supervisionada.
- Modelos: regressão linear simples e múltipla, Ridge, Lasso e polinomial de grau 2.
- O teste corresponde a 20% dos dados, com `random_state=42`.
- Outliers não são removidos automaticamente e o alvo não é transformado por defeito.

## 📊 PROMPT 1 — Análise Exploratória

```text
Cria `01_analise_exploratoria.py` para analisar `voos_telemetria.csv`.
Usa pathlib, pandas, numpy, matplotlib e seaborn. Não cries funções, `main` ou `if __name__ == "__main__":`.

Cria `artefactos/eda`, `artefactos/tabelas` e `artefactos/graficos`. Valida a existência do CSV e as colunas esperadas. Imprime dimensão, colunas, primeiras linhas, tipos, valores em falta, duplicados, estatísticas numéricas e frequências de `condicao_meteo`.

Analisa `duracao_voo_min`: média, mediana, desvio-padrão, quartis, skewness, limites IQR e potenciais outliers. Cria histograma com densidade e boxplot. Explica o impacto da assimetria e dos outliers, e por que o RMSE é mais sensível do que o MAE.

Cria histogramas, dispersões com linhas de tendência, matriz de correlação, heatmap, gráfico de contagens e boxplot por condição meteorológica. Verifica negativos, zeros inválidos, categorias inesperadas e riscos de data leakage. Não alteres o dataset.

Guarda PNG a 300 dpi, PDF dos gráficos principais, CSV das tabelas e `artefactos/eda/resumo_eda.md`. Usa comentários extensos e mensagens em português europeu.
```

## 🧹 PROMPT 2 — Pré-processamento

```text
Cria `02_preprocessamento.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pickle ou joblib, pandas, numpy e scikit-learn.

Valida o dataset, separa X e y e divide com 80% treino e 20% teste, `random_state=42`, sem embaralhar informação entre conjuntos. Explica por que `duracao_voo_min` nunca pode estar em X e por que não se estratifica diretamente um alvo contínuo.

Cria duas representações de `condicao_meteo`: OrdinalEncoder com Bom=0, Moderado=1, Adverso=2, e OneHotEncoder com `handle_unknown="ignore"`. Ajusta todos os objetos apenas no treino. Escala apenas variáveis numéricas com StandardScaler.

Guarda dados originais, matrizes ordinais, one-hot e escaladas, y_train/y_test, índices, nomes das features e preprocessadores em `artefactos/dados_processados` e `artefactos/preprocessamento`. Não removas outliers nem transformes o alvo automaticamente. Guarda `resumo_preprocessamento.md`.
```

## 🤖 PROMPT 3 — Treino dos Modelos

```text
Cria `03_treino_modelos.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, time, pickle/joblib, pandas, numpy e scikit-learn.

Carrega exclusivamente os dados processados. Treina: regressão linear simples só com `distancia_planeada`; regressão linear múltipla com one-hot; Ridge com alpha=1.0 e dados escalados; Lasso com alpha=0.1 e `max_iter` elevado; regressão polinomial de grau 2 com PolynomialFeatures e dados ajustados apenas no treino.

Regista parâmetros, tempos, observações e features. Gera previsões de treino e teste sem arredondar, valida NaN e alinhamento de índices e guarda modelos em `artefactos/modelos` e previsões em `artefactos/previsoes`. Não calcules métricas neste script.
```

## 📏 PROMPT 4 — Avaliação e Comparação

```text
Cria `04_avaliacao_modelos.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e sklearn.metrics.

Carrega e valida as previsões de teste. Calcula R², MAE, MSE e RMSE pela raiz quadrada do MSE. Guarda CSV com precisão total e Markdown com quatro casas, ordenado por RMSE.

Calcula erro médio com sinal, percentagens de subestimação e sobrestimação, maior erro de cada tipo e diferença percentual face ao modelo base. Explica R² negativo, MAE, RMSE, MSE, outliers e o custo operacional de subestimar ou sobrestimar a duração.

Seleciona o melhor modelo pelo menor RMSE e guarda métricas, análise de erros, `melhor_modelo.txt` e `resumo_avaliacao.md`.
```

## 🎯 PROMPT 5 — Previsto vs. Real

```text
Cria `05_previsto_vs_real.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy, matplotlib, seaborn e scikit-learn.

Seleciona automaticamente o modelo com menor RMSE. Cria dispersão com valores reais no eixo X, previstos no eixo Y, linha de identidade, escalas iguais, métricas no título e destaque dos cinco maiores erros.

Guarda PNG a 300 dpi, PDF, tabela dos maiores erros e interpretação em Markdown. Explica subestimação, sobrestimação, desempenho em voos curtos e longos e limitações do gráfico. Não treines modelos novamente.
```

## 📉 PROMPT 6 — Análise de Resíduos

```text
Cria `06_analise_residuos.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy, matplotlib, seaborn e scipy apenas se disponível.

Seleciona o melhor modelo e define `residuo = valor_real - valor_previsto`. Calcula estatísticas, skewness, percentagens de resíduos positivos/negativos e correlação entre erro absoluto e previsão.

Cria histograma com densidade e linha zero, resíduos contra previstos e resíduos absolutos contra previstos. Analisa centragem, caudas, outliers, não linearidade e heterocedasticidade, sem afirmar normalidade apenas pelo histograma.

Guarda PNG/PDF a 300 dpi, maiores resíduos em CSV/Markdown e `resumo_residuos.md`.
```

## 📝 PROMPT 7 — Relatório Automático

```text
Cria `07_gerar_relatorio.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e bibliotecas padrão.

Lê os artefactos existentes sem voltar a treinar. Gera `RELATORIO_FINAL.md` e uma cópia em `artefactos/relatorios`. Inclui introdução, dados, EDA, pré-processamento, data leakage, modelos, métricas, comparação com baseline, gráfico previsto vs. real, resíduos, coeficientes linear/Lasso, discussão, conclusões, limitações, recomendações e referências.

Usa caminhos relativos para imagens, data/hora, tabela de artefactos e “não disponível” quando faltar informação. Não inventes resultados.
```

## 🚀 PROMPT 8 — Orquestrador

```text
Cria `lab_orquestrador.py` sem funções, `main` ou bloco condicional.
Usa pathlib, subprocess, argparse, sys, time, logging e bibliotecas padrão.

Executa pela ordem 01 EDA, 02 pré-processamento, 03 treino, 04 avaliação, 05 gráfico, 06 resíduos e 07 relatório. Valida scripts e dataset, usa `sys.executable`, captura stdout/stderr, regista tempos em `artefactos/logs/execucao.log` e guarda resumo CSV/Markdown.

Aceita executar tudo, uma etapa ou intervalo, listar etapas, continuar após erro e repetir uma etapa. Não gera dados fictícios por defeito e termina com código de saída coerente.
```
