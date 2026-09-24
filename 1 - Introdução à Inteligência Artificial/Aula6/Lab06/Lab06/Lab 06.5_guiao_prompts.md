# Guião de Prompts para Lab 06.5 — Deep Learning: Regressão com um Neurónio

## 📚 Introdução ao Prompt Engineering

1. **Sê específico:** define dataset, alvo, arquitetura e métricas.
2. **Dá contexto:** explica a equivalência entre um neurónio linear e regressão linear.
3. **Pede exemplos:** exige tensores, losses, pesos e previsões.
4. **Itera:** valida cada etapa antes da seguinte.
5. **Estrutura a tarefa:** separa dados, PyTorch, scikit-learn e relatório.

## Assunções e Inferências

- Dataset: `voos_telemetria.csv`.
- Alvo: `duracao_voo_min`.
- Features: `distancia_planeada`, `carga_util_kg`, `altitude_media_m` e one-hot de `condicao_meteo`.
- Divisão: 70% treino, 15% validação e 15% teste, com seed fixa.
- Modelo PyTorch: `nn.Linear(n_features, 1)`, sem ativação.
- Perda: MSE; optimizer: Adam ou SGD, com hiperparâmetros explícitos.
- Comparação: regressão linear múltipla do scikit-learn com o mesmo pré-processamento.
- Métricas: MAE, MSE, RMSE e R².
- PyTorch e scikit-learn podem divergir ligeiramente por convergência e tolerâncias.
- Não há remoção automática de outliers nem transformação automática do alvo.

## 📊 PROMPT 1 — Análise Exploratória

```text
Cria `01_analise_exploratoria.py` para `voos_telemetria.csv`.
Usa pathlib, pandas, numpy, matplotlib e seaborn. Não cries funções, `main` ou `if __name__ == "__main__":`.

Cria `artefactos/eda`, `artefactos/tabelas` e `artefactos/graficos`. Valida a existência do CSV e as colunas esperadas. Imprime dimensão, primeiras linhas, tipos, valores em falta, duplicados e estatísticas.

Analisa `duracao_voo_min` com média, mediana, desvio-padrão, quartis, skewness, histograma, boxplot, limites IQR e potenciais outliers. Cria dispersões das variáveis numéricas contra o alvo, heatmap de correlação e gráficos da duração por condição meteorológica.

Verifica valores negativos, zeros inválidos, categorias inesperadas e risco de data leakage. Não alteres o dataset original. Guarda tabelas CSV, notas Markdown e gráficos PNG a 300 dpi/PDF. Fecha as figuras e escreve comentários e prints em português europeu.
```

## 🧹 PROMPT 2 — Pré-processamento e Tensores

```text
Cria `02_pre_processamento.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pickle ou joblib, pandas, numpy e scikit-learn.

Separa X e y, excluindo sempre `duracao_voo_min` de X. Aplica one-hot encoding a `condicao_meteo` com nomes estáveis. Divide sem leakage em 70% treino, 15% validação e 15% teste, usando seed=42.

Ajusta `StandardScaler` apenas em X_train e transforma validação/teste. Não ajustes o scaler em todos os dados. Guarda matrizes originais/processadas, índices, nomes das features, y e o scaler.

Converte as três matrizes X e os vetores y para tensores PyTorch com dtype float32. Imprime shapes, exemplos e confirma que treino, validação e teste têm o mesmo número de features. Guarda artefactos em `artefactos/dados` e `artefactos/modelos` e cria `resumo_pre_processamento.md`.
```

## 🧠 PROMPT 3 — Modelo PyTorch com um Neurónio

```text
Cria `03_modelo_pytorch.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, time, pickle/joblib, pandas, numpy e torch.

Carrega os tensores processados e fixa seeds. Define a classe `RegressaoNeuronio(nn.Module)` com `nn.Linear(n_features, 1)` e sem função de ativação. Implementa o loop de treino no corpo do script: model.train(), forward pass, MSE, backward pass, optimizer.step() e zero_grad().

Em cada época calcula loss de treino e validação com model.eval() e torch.no_grad(). Usa Adam ou SGD com learning rate explícito. Implementa early stopping apenas se ficar claro e guarda o melhor estado pela loss de validação.

Guarda o modelo, pesos, bias, histórico de losses, parâmetros, tempo e previsões de treino/validação/teste. Não arredondes previsões nem calcules ainda a tabela final de métricas. Explica no output a equação linear do neurónio.
```

## 📐 PROMPT 4 — Regressão Linear com scikit-learn

```text
Cria `04_regressao_sklearn.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pickle/joblib, pandas, numpy e scikit-learn.

Carrega exatamente os mesmos dados processados e os mesmos índices usados pelo PyTorch. Treina `LinearRegression` apenas com X_train e y_train. Gera previsões para treino, validação e teste sem voltar a ajustar o scaler.

Guarda o modelo, coeficientes, intercepto, nomes das features, previsões e parâmetros em `artefactos/modelos` e `artefactos/previsoes`. Imprime a equação e explica que `coef_` e `intercept_` estão na escala das features processadas.

Não calcules a comparação final neste script. Confirma que não existem NaN e que o número de previsões corresponde ao número de observações.
```

## 📏 PROMPT 5 — Avaliação e Comparação

```text
Cria `05_avaliacao_comparacao.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e sklearn.metrics.

Carrega as previsões de PyTorch e scikit-learn e calcula MAE, MSE, RMSE e R² em treino, validação e teste. Guarda uma tabela CSV com precisão total e uma versão Markdown com quatro casas.

Compara pesos e bias dos dois modelos, respeitando a mesma ordem e escala das features. Calcula diferenças absolutas e relativas. Explica que um neurónio sem ativação realiza `y = Xw + b`, a mesma operação da regressão linear, mas que resultados exatos dependem da convergência do treino.

Guarda `metricas_modelos.csv`, `comparacao_pesos.csv` e `resumo_comparacao.md`. Não afirmes causalidade a partir dos coeficientes.
```

## 📈 PROMPT 6 — Visualizações e Resíduos

```text
Cria `06_visualizacoes.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy, matplotlib e seaborn.

Cria a curva de loss de treino e validação do PyTorch, gráficos de valores reais vs. previstos no teste para ambos os modelos, comparação de coeficientes e histogramas/dispersões de resíduos.

Inclui linha de identidade nos gráficos previsto vs. real, linha zero nos resíduos e legendas claras. Calcula o resíduo como real - previsto e explica que resíduo positivo representa subestimação.

Guarda todos os gráficos em PNG a 300 dpi e PDF em `artefactos/graficos`. Fecha as figuras e escreve `interpretacao_graficos.md`, sem voltar a treinar os modelos.
```

## 📝 PROMPT 7 — Relatório Automático

```text
Cria `07_gerar_relatorio.py`, sem funções, `main` ou bloco condicional.
Usa pathlib, pandas, numpy e bibliotecas padrão.

Lê os artefactos sem voltar a treinar. Gera `RELATORIO_FINAL.md` e uma cópia em `artefactos/relatorios` com: objetivo, dados, EDA, pré-processamento, divisão dos dados, tensors, arquitetura, loop de treino, hiperparâmetros, curva de aprendizagem, comparação PyTorch/scikit-learn, métricas, coeficientes, resíduos, equivalência matemática, limitações e recomendações.

Inclui imagens com caminhos relativos, tabela de artefactos, data/hora e “não disponível” quando faltar informação. Explica diferenças numéricas, risco de overfitting, importância da escala e quando PyTorch é preferível ao scikit-learn.
```

## 🚀 PROMPT 8 — Orquestrador

```text
Cria `lab_orquestrador.py` sem funções, `main` ou bloco condicional.
Usa pathlib, subprocess, argparse, sys, time, logging e bibliotecas padrão.

Executa pela ordem 01 EDA, 02 pré-processamento, 03 PyTorch, 04 scikit-learn, 05 avaliação, 06 visualizações e 07 relatório. Valida dataset e scripts, usa sys.executable, captura stdout/stderr, regista tempos em `artefactos/logs/execucao.log` e guarda resumo CSV/Markdown.

Aceita executar tudo, uma etapa ou intervalo, listar etapas, continuar após erro e tentar novamente. Verifica dependências, não gera dados fictícios por defeito e termina com código de saída coerente.
```
