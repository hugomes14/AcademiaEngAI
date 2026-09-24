# Guião de Prompts para Lab 06.4 — Ensembles: Desempenho e Explicabilidade

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** define dados, alvo, modelos e métricas.
2. **Dá contexto:** explica o risco de incidentes e o desbalanceamento.
3. **Pede exemplos:** exige prints, gráficos e interpretações.
4. **Itera:** valida cada artefacto antes da etapa seguinte.
5. **Estrutura a tarefa:** separa preparação, treino, avaliação e XAI.

## Assunções e Inferências

- Dataset: `voos_pre_voo.csv`, o mesmo do Lab 06.2.
- Alvo: `incidente_reportado`, classificação binária 0/1.
- Features: `idade_aeronave_anos`, `horas_voo_desde_ultima_manutencao`, `previsao_turbulencia`, `tipo_missao` e `experiencia_piloto_anos`.
- A classe positiva é minoritária; F1, Recall e ROC-AUC são mais relevantes do que Accuracy isolada.
- Modelos: Árvore de Decisão, Random Forest e XGBoost; CatBoost é opcional.
- O melhor modelo é escolhido pelo maior F1 e, em empate, pelo maior ROC-AUC.
- XAI: importância de variáveis, importância por permutação e Partial Dependence Plots.
- XGBoost/CatBoost só devem ser usados quando estiverem instalados; o pipeline deve continuar sem eles, com aviso claro.

## 📊 PROMPT 1 — Análise Exploratória

```text
Cria `01_analise_exploratoria.py` para `voos_pre_voo.csv`.
Usa pathlib, pandas, numpy, matplotlib e seaborn. Não cries funções, `main` ou `if __name__ == "__main__":`.

Cria as pastas `artefactos/eda`, `artefactos/tabelas` e `artefactos/graficos`. Valida o ficheiro e todas as colunas esperadas. Imprime dimensão, primeiras linhas, tipos, valores em falta, duplicados, valores impossíveis e distribuição absoluta/percentual de `incidente_reportado`.

Analisa as variáveis numéricas por classe com histogramas, boxplots e estatísticas. Analisa `previsao_turbulencia` e `tipo_missao` por classe com countplots e tabelas de contingência. Cria heatmap de correlações numéricas e discute desbalanceamento, outliers e data leakage.

Guarda gráficos PNG a 300 dpi e PDF, tabelas CSV e `artefactos/eda/resumo_eda.md`. Usa comentários extensos, prints pedagógicos e fecha as figuras.
```

## 🧹 PROMPT 2 — Pré-processamento

```text
Cria `02_pre_processamento.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pickle/joblib, pandas, numpy e scikit-learn.

Separa X e y, excluindo `incidente_reportado` de X. Divide em treino/teste com 80/20, `random_state=42`, `shuffle=True` e `stratify=y`. Explica por que a estratificação protege a representação da classe minoritária.

Usa OneHotEncoder com `handle_unknown="ignore"` para as categóricas e mantém as numéricas. Cria um ColumnTransformer ajustado exclusivamente no treino. Guarda versões escaladas e não escaladas quando útil, os índices, nomes das features, y_train/y_test e os preprocessadores.

Valida valores em falta, categorias inesperadas, NaN nas matrizes e correspondência das classes. Guarda tudo em `artefactos/dados` e `artefactos/modelos`, com `resumo_pre_processamento.md`. Explica data leakage e não alteres o CSV original.
```

## 🌳 PROMPT 3 — Treino dos Modelos Ensemble

```text
Cria `03_treino_modelos_ensemble.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, time, pickle/joblib, pandas, numpy e scikit-learn. Usa xgboost e catboost apenas se estiverem instalados.

Carrega os dados processados e treina:
- Árvore de Decisão sem limite de profundidade como baseline;
- RandomForestClassifier com `n_estimators`, `random_state=42`, `class_weight="balanced"` e `oob_score=True`;
- XGBClassifier com pesos adequados para a classe minoritária e validação cruzada simples para escolher `n_estimators`;
- CatBoostClassifier opcional, com aviso se faltar a biblioteca.

Guarda modelos, previsões de classe, probabilidades/scores, parâmetros e tempos em `artefactos/modelos` e `artefactos/previsoes`. Não calcules métricas neste script. Explica viés-variância, overfitting da árvore, redução de variância no bagging e redução de viés no boosting.
```

## 📏 PROMPT 4 — Avaliação e Comparação

```text
Cria `04_avaliacao_modelos_ensemble.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e sklearn.metrics.

Valida todos os ficheiros de previsões e calcula Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, FPR e FNR. Usa `zero_division=0` quando necessário.

Cria tabela CSV com precisão total e Markdown com quatro casas. Ordena por F1 e usa ROC-AUC como desempate. Inclui treino/teste quando disponível para identificar overfitting, OOB score do Random Forest e comparação com resultados do Lab 06.2 se os artefactos existirem.

Guarda `metricas_ensemble.csv`, `metricas_ensemble.md`, `melhor_modelo.json` e `resumo_avaliacao.md`. Explica que Accuracy pode esconder falhas na classe positiva e discute custos de falsos positivos e falsos negativos.
```

## 🔍 PROMPT 5 — Importância das Variáveis e XAI

```text
Cria `05_importancia_variaveis.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy, matplotlib, seaborn e scikit-learn.

Carrega os modelos treinados e os nomes das features. Extrai `feature_importances_` da Árvore, Random Forest e XGBoost quando disponível. Calcula também importância por permutação no conjunto de teste, sem alterar o modelo.

Cria gráficos de barras ordenados, compara os rankings e identifica as três variáveis mais influentes. Guarda tabelas CSV, PNG a 300 dpi, PDF e uma interpretação Markdown.

Explica que importância não implica causalidade, que one-hot pode dividir a importância entre categorias e que a importância por permutação pode depender da correlação entre variáveis.
```

## 📈 PROMPT 6 — Partial Dependence e Interpretação

```text
Cria `06_partial_dependence.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy, matplotlib, seaborn e scikit-learn.

Seleciona automaticamente as uma ou duas features mais importantes do Random Forest ou XGBoost. Gera Partial Dependence Plots para a probabilidade da classe `incidente_reportado=1`, usando o pipeline completo e dados de teste sem alterar o modelo.

Quando a feature for categórica codificada, explica cuidadosamente o significado dos níveis. Mostra se a relação aparente aumenta, diminui ou não é monotónica. Guarda gráficos PNG/PDF a 300 dpi e `interpretacao_xai.md`.

Inclui um aviso de que PDP representa um efeito médio no conjunto e pode ser enganador perante interações ou extrapolação. Não uses linguagem causal.
```

## 📝 PROMPT 7 — Relatório Automático

```text
Cria `07_gerar_relatorio.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e bibliotecas padrão.

Lê os artefactos existentes sem voltar a treinar. Gera `RELATORIO_FINAL.md` e uma cópia em `artefactos/relatorios` com: introdução, dados, desbalanceamento, EDA, pipeline, modelos, viés-variância, OOB, tabela de métricas, comparação com Lab 06.2, importância de variáveis, permutação, PDP, análise de overfitting, custos FP/FN, limitações, conclusões e recomendações.

Inclui gráficos através de caminhos relativos, tabela de artefactos, data/hora e “não disponível” quando XGBoost/CatBoost ou outro resultado faltar. Não inventes resultados nem confundas associação com causalidade.
```

## 🚀 PROMPT 8 — Orquestrador

```text
Cria `lab_orquestrador.py` sem funções, `main` ou bloco condicional.
Usa pathlib, subprocess, argparse, sys, time, logging e bibliotecas padrão.

Executa pela ordem 01 EDA, 02 pré-processamento, 03 treino ensemble, 04 avaliação, 05 importância, 06 PDP e 07 relatório. Valida o dataset e os scripts, usa `sys.executable`, captura stdout/stderr, regista duração e erros em `artefactos/logs/execucao.log` e guarda resumo CSV/Markdown.

Aceita executar todas as etapas, uma etapa ou intervalo, listar etapas, continuar após erro e tentar novamente. Não gera dados fictícios por defeito. Dependências opcionais em falta devem produzir aviso e permitir relatório parcial. Termina com código de saída coerente.
```
