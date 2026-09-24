# Guião de Prompts para Lab 06.2 — Classificação: Prever o Risco de Incidentes em Voos

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** define alvo, classes, métricas e custos.
2. **Dá contexto:** explica que incidentes são a classe minoritária.
3. **Pede exemplos:** exige matrizes, métricas e interpretações.
4. **Itera:** valida cada etapa antes do treino seguinte.
5. **Estrutura a tarefa:** separa EDA, pipeline, treino e avaliação.

## Assunções e Inferências

- Dataset: `voos_pre_voo.csv`.
- Alvo: `incidente_reportado`, binário 0/1 e desbalanceado.
- Numéricas: `idade_aeronave_anos`, `horas_voo_desde_ultima_manutencao`, `experiencia_piloto_anos`.
- Categóricas: `previsao_turbulencia` e `tipo_missao`.
- O split é estratificado, temporalmente não aplicável ao dataset fornecido.
- Modelos: Regressão Logística, KNN, SVM linear, SVM RBF e Naive Bayes Gaussiano, incluindo versões balanceadas quando adequado.
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC e Specificity.
- Recall e F1 têm prioridade sobre Accuracy devido ao custo dos falsos negativos.

## 📊 PROMPT 1 — Análise Exploratória

```text
Cria `01_analise_exploratoria.py` para `voos_pre_voo.csv`.
Usa pathlib, pandas, numpy, matplotlib e seaborn. Não cries funções, `main` ou `if __name__ == "__main__":`.

Valida o CSV e as colunas. Imprime dimensão, primeiras linhas, tipos, valores em falta, duplicados, valores impossíveis e distribuição absoluta/percentual de `incidente_reportado`.

Analisa variáveis numéricas por classe com histogramas e boxplots, categóricas por classe com countplots e tabelas de contingência, e cria correlações numéricas. Discute desbalanceamento, possíveis relações e riscos de data leakage: só devem entrar no modelo informações disponíveis antes da descolagem.

Guarda gráficos PNG a 300 dpi e PDF, tabelas CSV e `artefactos/eda/notas_eda.md`. Fecha todas as figuras e escreve comentários e prints em português europeu.
```

## 🧹 PROMPT 2 — Pré-processamento

```text
Cria `02_pre_processamento.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pickle/joblib, pandas, numpy e scikit-learn.

Separa X e y, excluindo sempre `incidente_reportado` de X. Divide em treino/teste com 80/20, `random_state=42`, `shuffle=True` e `stratify=y`. Explica por que a estratificação é importante em classes desbalanceadas.

Usa OneHotEncoder com `handle_unknown="ignore"` para categóricas e StandardScaler apenas nas numéricas. Ajusta ColumnTransformer exclusivamente no treino e aplica-o ao treino/teste. Guarda versões escaladas e não escaladas para demonstrar o efeito da escala.

Guarda matrizes, índices, nomes das features, y_train/y_test, preprocessadores e `resumo_pre_processamento.md` em `artefactos`. Valida NaN, categorias e correspondência de classes.
```

## 🤖 PROMPT 3 — Treino dos Modelos

```text
Cria `03_treino_modelos.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, time, pickle/joblib, pandas, numpy e scikit-learn.

Carrega os artefactos processados e treina Regressão Logística, Regressão Logística com `class_weight="balanced"`, KNN, SVM linear, SVM RBF e Naive Bayes Gaussiano. Inclui versões balanceadas de SVM quando justificadas e versões sem escala de KNN/SVM apenas para comparação.

Guarda modelos, previsões de classe, scores/probabilidades quando disponíveis, tempos, parâmetros e nomes das features. Não calcules métricas neste script. Confirma que o teste não foi usado no ajuste.
```

## 📏 PROMPT 4 — Avaliação e Métricas

```text
Cria `04_avaliacao_modelos.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e sklearn.metrics.

Valida todas as previsões. Calcula Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, FPR e FNR, tratando corretamente o caso em que um modelo não fornece probabilidades.

Cria uma tabela CSV com precisão total e Markdown a quatro casas. Ordena por F1 e usa ROC-AUC como desempate. Explica por que Accuracy pode enganar, por que Recall é crítico para incidentes e os custos de falsos positivos e falsos negativos.

Guarda `metricas_classificacao.csv`, Markdown, JSON do melhor modelo e `resumo_avaliacao.md`.
```

## 🔲 PROMPT 5 — Matriz de Confusão

```text
Cria `05_matriz_confusao.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy, matplotlib, seaborn e scikit-learn.

Seleciona automaticamente o melhor modelo pelo maior F1 e, em empate, pelo maior ROC-AUC. Cria heatmap da matriz de confusão com contagens e percentagens. Para classificação binária mostra TN, FP, FN, TP, Recall, Specificity, FPR e FNR.

Interpreta os erros no contexto de segurança: um falso negativo pode deixar passar um incidente; um falso positivo pode consumir recursos e gerar alertas desnecessários. Guarda PNG/PDF a 300 dpi, JSON e Markdown.
```

## 📈 PROMPT 6 — Curvas ROC e Precision-Recall

```text
Cria `06_curvas_roc_pr.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy, matplotlib, seaborn e scikit-learn.

Carrega os scores dos modelos e cria uma curva ROC comparativa com AUC na legenda. Cria também uma curva Precision-Recall comparativa, pois a classe positiva é minoritária. Usa linhas de referência e títulos em português.

Guarda PNG a 300 dpi e PDF em `artefactos/graficos`. Explica que ROC-AUC mede separabilidade global e PR-AUC é mais informativa quando existem poucos positivos. Não seleciones manualmente o melhor modelo.
```

## 📝 PROMPT 7 — Relatório Automático

```text
Cria `07_gerar_relatorio.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e bibliotecas padrão.

Lê os artefactos existentes sem voltar a treinar. Gera `RELATORIO_FINAL.md` e uma cópia em `artefactos/relatorios`. Inclui introdução, dados, desbalanceamento, EDA, pré-processamento, data leakage, modelos, tabela de métricas, matriz de confusão, curvas ROC/PR, custos FP/FN, escolha do melhor modelo, limitações e recomendações sobre threshold, class_weight, tuning e validação cruzada estratificada.

Usa caminhos relativos para imagens, data/hora, tabela de artefactos e “não disponível” quando faltar informação. Não inventes resultados.
```

## 🚀 PROMPT 8 — Orquestrador

```text
Cria `lab_orquestrador.py` sem funções, `main` ou bloco condicional.
Usa pathlib, subprocess, argparse, sys, time, logging e bibliotecas padrão.

Executa pela ordem 01 EDA, 02 pré-processamento, 03 treino, 04 avaliação, 05 matriz de confusão, 06 curvas ROC/PR e 07 relatório. Valida dataset e scripts, usa `sys.executable`, captura stdout/stderr, regista tempos e erros em `execucao.log` e guarda resumo CSV/Markdown.

Aceita executar todas as etapas, uma etapa ou intervalo, listar etapas, continuar após erro e tentar novamente. Não gera dados fictícios por defeito e termina com código de saída coerente.
```
