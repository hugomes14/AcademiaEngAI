# -*- coding: utf-8 -*-
"""
Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos
Etapa 07: Relatório Automático em Markdown

Objetivo:
- Agregar resultados, gráficos e recomendações num ficheiro RELATORIO_FINAL.md.
- Produzir um relatório útil para revisão técnica e entrega do laboratório.
"""

from pathlib import Path
import json

import pandas as pd

print("=" * 80)
print("ETAPA 07 — RELATÓRIO AUTOMÁTICO")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "artefactos" / "dados"
EDA_DIR = BASE_DIR / "artefactos" / "eda"
METRICS_DIR = BASE_DIR / "artefactos" / "metricas"
GRAPH_DIR = BASE_DIR / "artefactos" / "graficos"
REPORT_DIR = BASE_DIR / "artefactos" / "relatorios"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

metrics_path = METRICS_DIR / "metricas_classificacao.csv"
best_path = METRICS_DIR / "melhor_modelo.json"
cm_path = METRICS_DIR / "matriz_confusao_melhor_modelo.json"
metadata_path = DATA_DIR / "metadata_pre_processamento.json"

for path in [metrics_path, best_path, metadata_path]:
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro obrigatório em falta para o relatório: {path}")

metrics_df = pd.read_csv(metrics_path)
best_info = json.loads(best_path.read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
cm_info = json.loads(cm_path.read_text(encoding="utf-8")) if cm_path.exists() else {}

# Tabela compacta para o relatório.
cols_for_report = [
    "modelo", "dados_usados", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "fp", "fn", "tp"
]
report_metrics = metrics_df[cols_for_report].copy()
for col in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
    report_metrics[col] = report_metrics[col].astype(float).round(4)

class_train = metadata.get("class_distribution_train", {})
class_test = metadata.get("class_distribution_test", {})

report = []
report.append("# RELATÓRIO FINAL — Lab 06.2 Classificação: Prever o Risco de Incidentes em Voos\n")

report.append("## 1. Introdução\n")
report.append(
    "Este relatório resume um pipeline de Machine Learning supervisionado para classificação binária. "
    "O objetivo é prever se um voo poderá ter um incidente de segurança (`incidente_reportado = 1`) "
    "com base em dados disponíveis antes da descolagem.\n"
)
report.append(
    "Como o problema envolve segurança, os erros não têm todos o mesmo custo. "
    "Um falso negativo significa classificar como seguro um voo que teve incidente, o que pode ser crítico. "
    "Por esse motivo, a análise valoriza métricas como Recall, F1, ROC-AUC e PR-AUC, e não apenas Accuracy.\n"
)

report.append("## 2. Dados e Análise Exploratória\n")
report.append(f"- Dataset usado: `{metadata.get('dataset', 'data/voos_pre_voo.csv')}`.")
report.append(f"- Variável-alvo: `{metadata.get('target', 'incidente_reportado')}`.")
report.append(f"- Variáveis numéricas: `{', '.join(metadata.get('numeric_cols', []))}`.")
report.append(f"- Variáveis ordinais: `{', '.join(metadata.get('ordinal_cols', []))}`.")
report.append(f"- Variáveis nominais: `{', '.join(metadata.get('nominal_cols', []))}`.")
report.append(f"- Distribuição no treino: `{class_train}`.")
report.append(f"- Distribuição no teste: `{class_test}`.\n")
report.append("![Distribuição da variável-alvo](../eda/01_distribuicao_alvo.png)\n")
report.append(
    "A análise exploratória confirma o desbalanceamento: a classe de incidente é minoritária. "
    "Isto torna a Accuracy insuficiente, porque um modelo que prevê sempre 'sem incidente' pode parecer bom, "
    "mas falhar exatamente os casos relevantes.\n"
)

report.append("## 3. Pipeline de Pré-processamento\n")
report.append(
    "O pipeline separou os dados em treino e teste com split estratificado, preservando a proporção das classes. "
    "A variável `previsao_turbulencia` foi tratada como ordinal, com a ordem Baixa < Média < Alta. "
    "A variável `tipo_missao` foi codificada com one-hot encoding, por não ter uma ordem natural. "
    "As variáveis numéricas e a variável ordinal codificada foram escalonadas com `StandardScaler`.\n"
)
report.append(
    "Para evitar data leakage, os transformadores foram ajustados apenas no conjunto de treino. "
    "O conjunto de teste recebeu apenas a transformação já aprendida no treino.\n"
)

report.append("## 4. Modelos Treinados\n")
report.append("Foram treinados os seguintes modelos:")
report.append("- Regressão Logística")
report.append("- Regressão Logística com `class_weight='balanced'`")
report.append("- K-Nearest Neighbors")
report.append("- SVM com kernel linear")
report.append("- SVM com kernel linear e `class_weight='balanced'`")
report.append("- SVM com kernel RBF")
report.append("- SVM com kernel RBF e `class_weight='balanced'`")
report.append("- Naïve Bayes Gaussiano")
report.append("- KNN e SVM RBF sem escalonamento, como comparação pedagógica sobre sensibilidade à escala.\n")

report.append("## 5. Resultados\n")
report.append(report_metrics.to_markdown(index=False))
report.append("\n")
report.append(
    f"O melhor modelo pelo critério principal **F1-Score** foi **{best_info['melhor_modelo']}**. "
    f"O F1 obtido foi **{best_info['f1']:.4f}**, com ROC-AUC de **{best_info['roc_auc']:.4f}**.\n"
)
report.append(
    "A métrica F1 foi escolhida porque equilibra Precision e Recall. "
    "Numa tarefa de risco de incidentes, Recall é especialmente importante, porque reduz falsos negativos. "
    "No entanto, Precision também importa, pois demasiados falsos positivos podem gerar alarmes operacionais desnecessários.\n"
)

report.append("## 6. Matriz de Confusão\n")
report.append("![Matriz de Confusão](../graficos/matriz_confusao_melhor_modelo.png)\n")
if cm_info:
    report.append(f"- TN: {cm_info.get('tn')} — voos sem incidente classificados como sem incidente.")
    report.append(f"- FP: {cm_info.get('fp')} — voos sem incidente classificados como incidentes.")
    report.append(f"- FN: {cm_info.get('fn')} — incidentes reais classificados como sem incidente.")
    report.append(f"- TP: {cm_info.get('tp')} — incidentes reais classificados como incidentes.")
    report.append(f"- Recall: {cm_info.get('recall'):.4f}.")
    report.append(f"- Specificity: {cm_info.get('specificity'):.4f}.\n")
report.append(
    "Os falsos negativos merecem atenção especial. Em contexto de segurança, um falso negativo pode significar "
    "que uma missão avança sem inspeção adicional apesar de existir risco.\n"
)

report.append("## 7. Curvas ROC e Precision-Recall\n")
report.append("![Curvas ROC](../graficos/curvas_roc_modelos.png)\n")
report.append("![Curvas Precision-Recall](../graficos/curvas_precision_recall_modelos.png)\n")
report.append(
    "A curva ROC permite comparar a separabilidade global dos modelos. "
    "A curva Precision-Recall é particularmente relevante neste dataset, porque a classe positiva é rara. "
    "Quando existe desbalanceamento, PR-AUC pode ser mais informativa do que ROC-AUC.\n"
)

report.append("## 8. Conclusões e Recomendações\n")
report.append(
    "O melhor modelo deve ser escolhido não só pela métrica agregada, mas também pelo custo operacional dos erros. "
    "Se o objetivo principal for minimizar incidentes não detetados, recomenda-se otimizar o threshold para aumentar Recall, "
    "mesmo que isso reduza Precision.\n"
)
report.append("Recomendações para continuação:")
report.append("- Testar thresholds diferentes de 0.5 para ajustar o equilíbrio Precision/Recall.")
report.append("- Comparar `class_weight='balanced'` com técnicas de reamostragem, como oversampling ou undersampling.")
report.append("- Fazer tuning de hiperparâmetros com validação cruzada estratificada.")
report.append("- Analisar outliers e casos mal classificados para perceber padrões operacionais.")
report.append("- Considerar novas variáveis preditoras disponíveis antes da descolagem, sem introduzir data leakage.")
report.append("- Rever o impacto de falsos positivos e falsos negativos com especialistas do domínio.\n")

report.append("## 9. Referências\n")
report.append("- Documentação oficial do scikit-learn sobre métricas de classificação.")
report.append("- Documentação oficial do scikit-learn sobre pipelines e pré-processamento.")
report.append("- Boas práticas de avaliação em datasets desbalanceados.\n")

report_path = REPORT_DIR / "RELATORIO_FINAL.md"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\nRelatório final gerado em:")
print(report_path)
print("\nEtapa 07 concluída com sucesso.")
