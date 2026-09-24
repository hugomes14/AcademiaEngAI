# Meta Prompt — Lab 06.5: Deep Learning — Regressão com um Neurónio

## QUEM ÉS

És especialista em ensino prático de Machine Learning, Deep Learning e Prompt Engineering. Cria um GUIÃO DE PROMPTS completo, executável e pedagógico para o Lab 06.5, em português europeu. O público sabe executar scripts Python, mas está a aprender PyTorch.

Em cada prompt exige comentários abundantes, prints informativos, nomes claros e código sequencial. Não permitas novas funções, uma função `main` ou o bloco `if __name__ == "__main__":` nos scripts produzidos.

## ENTRADA — BRIEF DO LABORATÓRIO

O laboratório demonstra que uma rede neural com um único neurónio linear é matematicamente equivalente à regressão linear múltipla.

Dataset: `voos_telemetria.csv`, com:

- `distancia_planeada`;
- `carga_util_kg`;
- `altitude_media_m`;
- `condicao_meteo` (`Bom`, `Moderado`, `Adverso`);
- `duracao_voo_min`, variável-alvo contínua.

Objetivos:

1. Explorar e validar os dados.
2. Aplicar one-hot encoding e padronização sem data leakage.
3. Separar os dados em treino 70%, validação 15% e teste 15%.
4. Converter features e alvo para tensores PyTorch.
5. Construir um modelo `nn.Linear` com uma saída e sem função de ativação.
6. Implementar o loop de treino com forward pass, MSE, backward pass e atualização dos pesos.
7. Monitorizar loss de treino e validação e permitir early stopping.
8. Treinar uma regressão linear equivalente com scikit-learn.
9. Comparar MAE, RMSE, MSE, R², pesos e bias dos dois modelos.
10. Criar curva de aprendizagem, previsto vs. real, comparação de coeficientes e resíduos.

## REGRAS GERAIS

- Língua: português europeu.
- Bibliotecas: pathlib, pandas, numpy, matplotlib, seaborn, scikit-learn, torch e pickle/joblib.
- Fixar sementes de NumPy, PyTorch e restantes fontes de aleatoriedade.
- Usar `StandardScaler` ajustado apenas no treino.
- Não transformar o alvo sem explicar a transformação inversa.
- Não calcular coeficientes em escalas incompatíveis sem documentar a conversão.
- Guardar modelos, scalers, tensores/matrizes, previsões, tabelas, gráficos e relatório.
- Usar PNG a pelo menos 300 dpi e PDF nos gráficos principais.
- Fechar todas as figuras.
- Não inventar resultados nem remover outliers automaticamente.
- Explicar por que resultados PyTorch e scikit-learn podem não ser exatamente iguais: convergência, learning rate, épocas, inicialização, tolerância e regularização.

## ARTEFACTOS ESPERADOS

- `artefactos/eda/`;
- `artefactos/dados/`;
- `artefactos/modelos/`;
- `artefactos/previsoes/`;
- `artefactos/graficos/`;
- `artefactos/tabelas/`;
- `artefactos/relatorios/`;
- `RELATORIO_FINAL.md`.

## TAREFA

Produz apenas o documento final chamado:

`Guião de Prompts para Lab 06.5 — Deep Learning: Regressão com um Neurónio`

O documento deve conter exatamente:

1. Título.
2. `📚 Introdução ao Prompt Engineering`, com cinco princípios.
3. `Assunções e Inferências`.
4. Oito secções de prompts encadeados.

Cada prompt deve conter:

- cabeçalho com emoji;
- bloco `O que vais aprender` com 3 a 5 pontos;
- bloco de código com o prompt pronto a copiar;
- nome do ficheiro a criar;
- bibliotecas permitidas;
- requisitos técnicos concretos;
- checklist `Após receber o código`.

## PROMPTS OBRIGATÓRIOS

1. `01_analise_exploratoria.py`: qualidade dos dados, distribuição do alvo, skewness, outliers, correlações e gráficos.
2. `02_pre_processamento.py`: limpeza, one-hot, divisão 70/15/15, scaler ajustado só no treino, matrizes e tensores.
3. `03_modelo_pytorch.py`: classe `RegressaoNeuronio(nn.Module)`, tensors, seed, optimizer, loss e loop de treino/validação.
4. `04_regressao_sklearn.py`: regressão linear múltipla com as mesmas features e o mesmo split, sem usar o teste no ajuste.
5. `05_avaliacao_comparacao.py`: MAE, MSE, RMSE e R², comparação de pesos/bias e análise da equivalência matemática.
6. `06_visualizacoes.py`: loss treino/validação, previsto vs. real, coeficientes PyTorch/scikit-learn e resíduos.
7. `07_gerar_relatorio.py`: relatório Markdown com todos os resultados, decisões, limitações e recomendações.
8. `lab_orquestrador.py`: execução sequencial, logs, validações, argumentos, tratamento de erros e resumo.

## RESULTADO ESPERADO

O guião deve ensinar que:

`y = w1*x1 + ... + wn*xn + b`

é a mesma forma matemática da regressão linear múltipla. Deve distinguir equivalência arquitetural de igualdade numérica das previsões e evitar interpretações causais dos coeficientes.
