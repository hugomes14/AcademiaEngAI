# Lab 06.3 — Séries Temporais
# Script 07 — Relatório Automático em Markdown
# Objetivo: criar RELATORIO_FINAL.md com resultados, imagens e recomendações.

from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
TAB_DIR = ARTEFACTOS_DIR / "tabelas"
REL_DIR = ARTEFACTOS_DIR / "relatorios"

REL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SCRIPT 07 — RELATÓRIO AUTOMÁTICO")
print("=" * 90)

metricas = pd.read_csv(TAB_DIR / "metricas_modelos.csv")
with open(TAB_DIR / "melhores_modelos.json", "r", encoding="utf-8") as f:
    melhores = json.load(f)

metricas_formatadas = metricas.copy()
for col in ["MAE", "MSE", "RMSE", "MAPE_percent", "sMAPE_percent", "R2"]:
    metricas_formatadas[col] = metricas_formatadas[col].map(lambda x: f"{x:.4f}")

tabela_md = metricas_formatadas.to_markdown(index=False)

relatorio = f"""# RELATÓRIO FINAL — Lab 06.3: Séries Temporais — Previsão de procura e consumo

## 1. Introdução

Este relatório resume a resolução do laboratório de séries temporais com dois cenários:

1. **Voltagem de bateria**, medida minuto a minuto.
2. **Número de missões diárias**, medido por dia.

O objetivo foi comparar baselines, modelos de Machine Learning com *lags* e calendário, e modelos estatísticos simples para prever valores futuros.

## 2. Dados utilizados

### Série 1 — Voltagem

- Ficheiro: `dados/voltagem_bateria.csv`
- Coluna temporal: `timestamp`
- Alvo: `voltagem`
- Frequência esperada: minuto a minuto

### Série 2 — Missões diárias

- Ficheiro: `dados/missoes_diarias.csv`
- Coluna temporal: `data`
- Alvo: `num_missoes`
- Frequência esperada: diária

## 3. Análise exploratória

A análise exploratória verificou:

- formato das datas;
- frequência temporal;
- lacunas no índice temporal;
- duplicados;
- distribuição da variável alvo;
- tendência e sazonalidade;
- autocorrelação através de ACF e PACF;
- estacionaridade através do teste ADF.

Imagens principais:

### Voltagem

![Série temporal da voltagem](../imagens/01_voltagem_serie_temporal.png)

![Decomposição da voltagem](../imagens/01_voltagem_decomposicao.png)

![ACF da voltagem](../imagens/01_voltagem_acf.png)

### Missões

![Série temporal das missões](../imagens/01_missoes_serie_temporal.png)

![Decomposição das missões](../imagens/01_missoes_decomposicao.png)

![ACF das missões](../imagens/01_missoes_acf.png)

## 4. Pipeline e engenharia de features

Para evitar *data leakage*, o split foi temporal:

- treino: primeiros 80% dos registos;
- teste: últimos 20% dos registos.

Foram criadas features como:

- variáveis de calendário;
- codificações cíclicas para horas, dias da semana e meses;
- *lags*;
- médias móveis;
- desvios-padrão móveis.

O escalonamento foi ajustado apenas no treino e aplicado depois ao teste.

## 5. Modelos treinados

Foram avaliados:

- Baseline de Persistência;
- Regressão Linear;
- Ridge;
- Random Forest;
- Suavização Exponencial;
- Prophet, apenas se disponível no ambiente.

O baseline é muito importante em séries temporais: um modelo só deve ser considerado útil se superar uma regra simples como “o próximo valor será semelhante ao último”.

## 6. Resultados

{tabela_md}

## 7. Melhores modelos

### Voltagem

- Melhor modelo por RMSE: `{melhores["voltagem"]["modelo"]}`
- RMSE: {melhores["voltagem"]["RMSE"]:.4f}

![Previsão vs Real — Voltagem](../imagens/05_voltagem_previsao_vs_real.png)

### Missões

- Melhor modelo por RMSE: `{melhores["missoes"]["modelo"]}`
- RMSE: {melhores["missoes"]["RMSE"]:.4f}

![Previsão vs Real — Missões](../imagens/05_missoes_previsao_vs_real.png)

## 8. Análise de resíduos

A análise de resíduos ajuda a perceber se o modelo deixou padrões temporais por capturar.

### Voltagem

![Histograma dos resíduos — Voltagem](../imagens/06_voltagem_residuos_histograma.png)

![Resíduos no tempo — Voltagem](../imagens/06_voltagem_residuos_tempo.png)

![ACF dos resíduos — Voltagem](../imagens/06_voltagem_residuos_acf.png)

### Missões

![Histograma dos resíduos — Missões](../imagens/06_missoes_residuos_histograma.png)

![Resíduos no tempo — Missões](../imagens/06_missoes_residuos_tempo.png)

![ACF dos resíduos — Missões](../imagens/06_missoes_residuos_acf.png)

## 9. Conclusões

A escolha do melhor modelo deve considerar:

- menor RMSE, quando erros grandes são especialmente penalizadores;
- MAE, quando se quer uma leitura direta do erro médio;
- MAPE/sMAPE, quando o erro percentual é relevante;
- capacidade de superar o baseline;
- estabilidade dos resíduos;
- interpretabilidade.

Para a série de voltagem, modelos com lags curtos tendem a ser relevantes, pois valores recentes costumam explicar bem o comportamento imediato.

Para missões diárias, features semanais e sazonais são especialmente úteis, porque a procura operacional pode variar conforme o dia da semana e o período do ano.

## 10. Recomendações

Próximos passos sugeridos:

1. Testar mais *lags* e janelas móveis.
2. Experimentar SARIMA ou SARIMAX com sazonalidade explícita.
3. Instalar e testar Prophet quando o ambiente permitir.
4. Avaliar *walk-forward validation* para simular previsões em produção.
5. Analisar feriados, campanhas, manutenção ou eventos operacionais como variáveis externas.
6. Comparar RMSE com MAE para perceber o impacto de picos e valores extremos.
7. Verificar a ACF dos resíduos: autocorrelação persistente indica estrutura temporal por modelar.

## 11. Referências

- `pandas` — manipulação de dados temporais.
- `scikit-learn` — modelos de regressão e métricas.
- `statsmodels` — decomposição, ACF/PACF e suavização exponencial.
- `matplotlib` e `seaborn` — visualização.
"""

(REL_DIR / "RELATORIO_FINAL.md").write_text(relatorio, encoding="utf-8")

print(f"[OK] Relatório guardado em: {REL_DIR / 'RELATORIO_FINAL.md'}")
print("=" * 90)
