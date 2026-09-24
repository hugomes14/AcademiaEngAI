# 07_relatorio_automatico.py
# Lab 06.4 — Modelos Ensemble
# Etapa: Relatório automático em Markdown

from pathlib import Path
import json

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
EDA_DIR = BASE_DIR / "artefactos" / "eda"
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
MODEL_DIR = BASE_DIR / "artefactos" / "modelos"
METRIC_DIR = BASE_DIR / "artefactos" / "metricas"
FIG_DIR = BASE_DIR / "artefactos" / "figuras"
REPORT_DIR = BASE_DIR / "artefactos" / "relatorios"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.4 — ETAPA 7: RELATÓRIO AUTOMÁTICO")
print("=" * 80)

metricas = pd.read_csv(METRIC_DIR / "metricas_modelos.csv")
melhor = json.loads((METRIC_DIR / "melhor_modelo.json").read_text(encoding="utf-8"))

distribuicao_alvo = pd.read_csv(EDA_DIR / "distribuicao_alvo.csv")
tempos = pd.read_csv(METRIC_DIR / "tempos_treino.csv") if (METRIC_DIR / "tempos_treino.csv").exists() else pd.DataFrame()

oob_texto = ""
if (METRIC_DIR / "random_forest_oob_score.txt").exists():
    oob_texto = (METRIC_DIR / "random_forest_oob_score.txt").read_text(encoding="utf-8").strip()

imagens = {
    "Distribuição do alvo": "../figuras/eda_distribuicao_alvo.png",
    "Correlação": "../figuras/eda_heatmap_correlacoes.png",
    "Matriz de Confusão": "../figuras/matriz_confusao_melhor_modelo.png",
    "Curvas ROC": "../figuras/curvas_roc_modelos.png",
    "Curvas Precision-Recall": "../figuras/curvas_precision_recall_modelos.png",
    "Árvore de Decisão": "../figuras/arvore_decisao_primeiros_niveis.png",
    "PDP": "../figuras/pdp_variaveis_importantes.png",
}

importancia_paths = sorted(METRIC_DIR.glob("importancia_variaveis_*.md"))
seccao_importancias = ""
for path in importancia_paths:
    if path.name == "importancia_variaveis_todos_modelos.md":
        continue
    seccao_importancias += f"\n### {path.stem.replace('importancia_variaveis_', '')}\n\n"
    seccao_importancias += path.read_text(encoding="utf-8") + "\n"

relatorio = f"""# RELATÓRIO FINAL — Lab 06.4: Modelos Ensemble

## 1. Introdução

Este relatório apresenta a resolução do laboratório de métodos ensemble para previsão de
incidentes em voos. O problema é uma tarefa de **classificação binária**:

- Classe 0: voo sem incidente reportado.
- Classe 1: voo com incidente reportado.

O objetivo é comparar uma Árvore de Decisão, Random Forest e modelos de boosting,
incluindo XGBoost quando disponível, e usar técnicas de explicabilidade para compreender
as previsões.

## 2. Dataset

O dataset utilizado foi `voos_pre_voo.csv`.

### Distribuição da variável-alvo

{distribuicao_alvo.to_markdown(index=False)}

![Distribuição do alvo]({imagens["Distribuição do alvo"]})

O dataset está desbalanceado. A classe positiva é minoritária, por isso a Accuracy deve
ser interpretada com cuidado.

## 3. Análise Exploratória

Foram analisadas:

- variáveis numéricas por classe;
- variáveis categóricas por classe;
- correlações entre variáveis numéricas e alvo;
- desbalanceamento da variável-alvo.

![Correlação]({imagens["Correlação"]})

## 4. Pré-processamento

O pipeline aplicou:

- split treino/teste estratificado;
- `StandardScaler` nas variáveis numéricas;
- encoding ordinal em `previsao_turbulencia`, com ordem Baixa < Média < Alta;
- one-hot encoding em `tipo_missao`;
- `fit` apenas no treino, para evitar data leakage.

## 5. Modelos

Foram treinados:

- `DecisionTreeClassifier`, como baseline e demonstração de potencial overfitting;
- `RandomForestClassifier`, como bagging;
- `XGBClassifier`, como boosting, se disponível;
- `GradientBoostingClassifier`, como alternativa se o XGBoost não estiver disponível;
- `CatBoostClassifier`, se disponível.

### Random Forest OOB

{oob_texto if oob_texto else "O OOB score não foi gerado ou o modelo Random Forest não foi treinado."}

### Tempos de treino

{tempos.to_markdown(index=False) if not tempos.empty else "Tempos de treino não disponíveis."}

## 6. Resultados

{metricas.round(4).to_markdown(index=False)}

### Melhor modelo

Pelo critério principal **F1-Score**, o melhor modelo foi:

- Modelo: `{melhor["modelo"]}`
- F1: {melhor["f1"]:.4f}
- ROC-AUC: {melhor["roc_auc"]:.4f}
- Recall: {melhor["recall"]:.4f}
- Precision: {melhor["precision"]:.4f}

## 7. Matriz de Confusão

![Matriz de Confusão]({imagens["Matriz de Confusão"]})

Os falsos negativos são especialmente críticos porque representam incidentes reais que
não foram assinalados pelo modelo. Os falsos positivos geram custos operacionais, mas,
num cenário de segurança, podem ser mais aceitáveis do que falhar incidentes.

## 8. Curvas ROC e Precision-Recall

![Curvas ROC]({imagens["Curvas ROC"]})

![Curvas Precision-Recall]({imagens["Curvas Precision-Recall"]})

A curva Precision-Recall é particularmente útil neste problema, porque a classe positiva
é rara.

## 9. Explicabilidade

### Árvore de Decisão

![Árvore de Decisão]({imagens["Árvore de Decisão"]})

A árvore isolada permite interpretação visual, mas pode apresentar alta variância e
sobreajuste quando não tem limite de profundidade.

### Importância de Variáveis

{seccao_importancias if seccao_importancias else "Não foram encontradas tabelas de importância de variáveis."}

### Partial Dependence Plots

![Partial Dependence Plots]({imagens["PDP"]})

Os PDP ajudam a perceber como as variáveis mais relevantes influenciam a probabilidade
prevista de incidente.

## 10. Conclusões

Os modelos ensemble permitem melhorar a robustez face a uma árvore isolada:

- Random Forest reduz variância ao combinar várias árvores.
- Boosting tenta reduzir erro sequencialmente, ao corrigir falhas de modelos anteriores.
- As métricas F1, Recall, ROC-AUC e PR-AUC são mais adequadas do que Accuracy para este
  cenário desbalanceado.

## 11. Recomendações

- Afinar hiperparâmetros com `GridSearchCV` ou `RandomizedSearchCV`.
- Testar thresholds diferentes de 0.5 para aumentar Recall ou Precision conforme o custo operacional.
- Avaliar reamostragem ou `class_weight` com mais detalhe.
- Usar SHAP, quando disponível, para explicabilidade local.
- Comparar os resultados com os modelos simples do Lab 06.2.
"""

output = REPORT_DIR / "RELATORIO_FINAL.md"
output.write_text(relatorio, encoding="utf-8")

print(f"\nRelatório gerado em: {output}")
print("\nEtapa 7 concluída com sucesso.")
