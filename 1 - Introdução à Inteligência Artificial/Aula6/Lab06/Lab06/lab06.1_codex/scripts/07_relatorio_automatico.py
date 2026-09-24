# 07_relatorio_automatico.py
# Lab 06.1 - Regressão — Estimar a Duração de Voo
# Objetivo: gerar um relatório final em Markdown com resultados, gráficos e recomendações.

from pathlib import Path
import pandas as pd

# -----------------------------
# Configuração de caminhos
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
TAB_DIR = OUTPUT_DIR / "tabelas"
NOTE_DIR = OUTPUT_DIR / "notas"
FIG_DIR = OUTPUT_DIR / "figuras"
REPORT_PATH = OUTPUT_DIR / "RELATORIO_FINAL.md"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("07 - RELATÓRIO AUTOMÁTICO")
print("=" * 80)

metrics_path = TAB_DIR / "04_metricas_modelos.csv"
metrics_md_path = TAB_DIR / "04_metricas_modelos.md"
coefficients_path = TAB_DIR / "03_coeficientes_modelos_lineares.csv"
residual_stats_path = TAB_DIR / "06_estatisticas_residuos.csv"
eda_notes_path = NOTE_DIR / "01_notas_eda.md"
preprocess_notes_path = NOTE_DIR / "02_notas_preprocessamento.md"
evaluation_notes_path = NOTE_DIR / "04_notas_avaliacao.md"
residual_notes_path = NOTE_DIR / "06_notas_residuos.md"

required_files = [metrics_path, metrics_md_path]
for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Ficheiro necessário não encontrado: {file_path}. Corre os scripts anteriores.")

metrics_df = pd.read_csv(metrics_path).sort_values(by=["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
best_model = metrics_df.iloc[0]["modelo"]
best_rmse = metrics_df.iloc[0]["RMSE"]
best_mae = metrics_df.iloc[0]["MAE"]
best_r2 = metrics_df.iloc[0]["R2"]
metrics_markdown = metrics_md_path.read_text(encoding="utf-8")

if coefficients_path.exists():
    coefficients_df = pd.read_csv(coefficients_path)
    coefficients_markdown = coefficients_df.to_markdown(index=False)
else:
    coefficients_markdown = "_Coeficientes não disponíveis._"

if residual_stats_path.exists():
    residual_stats_df = pd.read_csv(residual_stats_path)
    residual_stats_markdown = residual_stats_df.to_markdown(index=False)
else:
    residual_stats_markdown = "_Estatísticas de resíduos não disponíveis._"

eda_notes = eda_notes_path.read_text(encoding="utf-8") if eda_notes_path.exists() else "_Notas de EDA não disponíveis._"
preprocess_notes = preprocess_notes_path.read_text(encoding="utf-8") if preprocess_notes_path.exists() else "_Notas de pré-processamento não disponíveis._"
evaluation_notes = evaluation_notes_path.read_text(encoding="utf-8") if evaluation_notes_path.exists() else "_Notas de avaliação não disponíveis._"
residual_notes = residual_notes_path.read_text(encoding="utf-8") if residual_notes_path.exists() else "_Notas de resíduos não disponíveis._"

report_lines = []
report_lines.append("# Relatório Final — Lab 06.1: Regressão — Estimar a Duração de Voo\n")
report_lines.append("## 1. Introdução\n")
report_lines.append(
    "Este relatório resume a construção e avaliação de modelos de regressão para estimar "
    "a duração total de um voo com base em variáveis conhecidas antes da descolagem: "
    "distância planeada, carga útil, altitude média e condição meteorológica.\n"
)
report_lines.append(
    "A variável-alvo é `duracao_voo_min`, um valor contínuo em minutos. "
    "Por isso, trata-se de uma tarefa de aprendizagem supervisionada de regressão.\n"
)

report_lines.append("## 2. Análise Exploratória\n")
report_lines.append(eda_notes + "\n")
report_lines.append("### Figuras principais da EDA\n")
report_lines.append("![Distribuição da duração do voo](figuras/01_distribuicao_duracao_voo.png)\n")
report_lines.append("![Boxplot da duração do voo](figuras/01_boxplot_duracao_voo.png)\n")
report_lines.append("![Correlação entre variáveis numéricas](figuras/01_heatmap_correlacao.png)\n")
report_lines.append("![Condição meteorológica vs duração](figuras/01_boxplot_condicao_meteo_vs_duracao.png)\n")

report_lines.append("## 3. Pipeline de Pré-processamento\n")
report_lines.append(preprocess_notes + "\n")
report_lines.append(
    "O ponto crítico do pré-processamento foi evitar data leakage: o escalonamento foi ajustado apenas "
    "no conjunto de treino e aplicado depois ao treino e ao teste. Esta regra mantém o teste como simulação "
    "de dados futuros.\n"
)

report_lines.append("## 4. Modelos Treinados\n")
report_lines.append("Foram treinados os seguintes modelos:\n")
report_lines.append("- Regressão Linear Simples com `distancia_planeada`.\n")
report_lines.append("- Regressão Linear Múltipla com todas as variáveis preditoras.\n")
report_lines.append("- Ridge com regularização L2.\n")
report_lines.append("- Lasso com regularização L1.\n")
report_lines.append("- Regressão Polinomial de grau 2.\n")
report_lines.append(
    "Ridge, Lasso e Regressão Polinomial usaram dados escalados, porque estes modelos são sensíveis "
    "à escala das features.\n"
)

report_lines.append("## 5. Resultados e Métricas\n")
report_lines.append(metrics_markdown + "\n")
report_lines.append(evaluation_notes + "\n")
report_lines.append(
    f"O melhor modelo pelo critério RMSE foi `{best_model}`, com RMSE de **{best_rmse:.4f} minutos**, "
    f"MAE de **{best_mae:.4f} minutos** e R² de **{best_r2:.4f}**.\n"
)

report_lines.append("## 6. Interpretação de Coeficientes\n")
report_lines.append(
    "A tabela seguinte reúne coeficientes da regressão linear múltipla e do Lasso. "
    "Na regressão linear múltipla, os coeficientes estão nas unidades originais das variáveis. "
    "No Lasso, os coeficientes estão na escala transformada, porque o modelo usa features escaladas.\n"
)
report_lines.append(coefficients_markdown + "\n")
report_lines.append(
    "Coeficientes próximos de zero no Lasso sugerem menor utilidade preditiva após regularização. "
    "No entanto, a decisão de remover uma variável deve considerar também conhecimento de domínio e estabilidade em validações adicionais.\n"
)

report_lines.append("## 7. Gráfico Previsto vs. Real\n")
report_lines.append("![Previsto vs. Real](figuras/05_previsto_vs_real_melhor_modelo.png)\n")
report_lines.append(
    "Pontos próximos da linha diagonal indicam boas previsões. Pontos abaixo da linha indicam "
    "sub-estimação; pontos acima indicam sobre-estimação. Em planeamento de voos, a sub-estimação "
    "pode ser particularmente crítica por afetar margens operacionais.\n"
)

report_lines.append("## 8. Análise de Resíduos\n")
report_lines.append(residual_notes + "\n")
report_lines.append("### Estatísticas dos resíduos\n")
report_lines.append(residual_stats_markdown + "\n")
report_lines.append("![Histograma dos resíduos](figuras/06_histograma_residuos_melhor_modelo.png)\n")
report_lines.append("![Resíduos vs. previstos](figuras/06_residuos_vs_previstos_melhor_modelo.png)\n")

report_lines.append("## 9. Conclusões e Recomendações\n")
report_lines.append(
    f"O modelo recomendado nesta execução é `{best_model}`, porque obteve o menor RMSE no conjunto de teste. "
    "Como o RMSE penaliza mais os erros grandes, esta escolha favorece um modelo que controla melhor desvios elevados.\n"
)
report_lines.append("\nRecomendações para trabalho futuro:\n")
report_lines.append("- Testar transformação do alvo se a distribuição de `duracao_voo_min` apresentar assimetria forte.\n")
report_lines.append("- Afinar hiperparâmetros de Ridge, Lasso e Polinomial com validação cruzada.\n")
report_lines.append("- Analisar outliers com cuidado antes de os remover. Alguns podem representar voos raros, mas válidos.\n")
report_lines.append("- Criar novas features de domínio, por exemplo categorias de distância ou indicadores de carga elevada.\n")
report_lines.append("- Avaliar o custo operacional de sub-estimar vs. sobre-estimar a duração de voo.\n")
report_lines.append("- Repetir a avaliação com mais folds ou dados de períodos distintos para testar generalização temporal.\n")

report_lines.append("## 10. Referências\n")
report_lines.append("- Documentação do pandas.\n")
report_lines.append("- Documentação do scikit-learn sobre regressão, métricas e pré-processamento.\n")
report_lines.append("- Boas práticas de avaliação de modelos de regressão e prevenção de data leakage.\n")

REPORT_PATH.write_text("".join(report_lines), encoding="utf-8")

print(f"Relatório gerado em: {REPORT_PATH}")
print("\n07 - Relatório automático concluído com sucesso.")
