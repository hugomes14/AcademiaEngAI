# 06_analise_residuos.py
# Lab 06.1 - Regressão — Estimar a Duração de Voo
# Objetivo: analisar os resíduos do melhor modelo com histograma e dispersão resíduos vs. previstos.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Configuração de caminhos
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
PRED_DIR = OUTPUT_DIR / "previsoes"
TAB_DIR = OUTPUT_DIR / "tabelas"
FIG_DIR = OUTPUT_DIR / "figuras"
NOTE_DIR = OUTPUT_DIR / "notas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("06 - ANÁLISE DE RESÍDUOS")
print("=" * 80)

metrics_path = TAB_DIR / "04_metricas_modelos.csv"
if not metrics_path.exists():
    raise FileNotFoundError(f"Métricas não encontradas em {metrics_path}. Corre primeiro o script 04.")

metrics_df = pd.read_csv(metrics_path)
metrics_df = metrics_df.sort_values(by=["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
best_model = metrics_df.iloc[0]["modelo"]

pred_path = PRED_DIR / f"previsoes_{best_model}.csv"
if not pred_path.exists():
    raise FileNotFoundError(f"Previsões do melhor modelo não encontradas em {pred_path}.")

pred_df = pd.read_csv(pred_path)
pred_df["residuo"] = pred_df["y_real"] - pred_df["y_previsto"]
pred_df["erro_absoluto"] = pred_df["residuo"].abs()
pred_df["erro_quadratico"] = pred_df["residuo"] ** 2
pred_df.to_csv(TAB_DIR / "06_residuos_melhor_modelo.csv", index=False)

residuals = pred_df["residuo"]
residual_stats = pd.DataFrame({
    "metrica": ["modelo", "media", "mediana", "desvio_padrao", "minimo", "maximo", "skewness", "erro_absoluto_medio"],
    "valor": [
        best_model,
        residuals.mean(),
        residuals.median(),
        residuals.std(ddof=1),
        residuals.min(),
        residuals.max(),
        residuals.skew(),
        pred_df["erro_absoluto"].mean()
    ]
})
residual_stats.to_csv(TAB_DIR / "06_estatisticas_residuos.csv", index=False)

print(f"Melhor modelo analisado: {best_model}")
print("\nEstatísticas dos resíduos:")
print(residual_stats)

# -----------------------------
# Histograma dos resíduos
# -----------------------------
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor="black", alpha=0.8)
plt.axvline(0, linestyle="--", linewidth=2)
plt.title(f"Distribuição dos resíduos — {best_model}")
plt.xlabel("Resíduo = real - previsto (minutos)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_histograma_residuos_melhor_modelo.png", dpi=300)
plt.savefig(FIG_DIR / "06_histograma_residuos_melhor_modelo.pdf")
plt.close()

# -----------------------------
# Resíduos vs. valores previstos
# -----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(pred_df["y_previsto"], residuals, alpha=0.75)
plt.axhline(0, linestyle="--", linewidth=2)
plt.title(f"Resíduos vs. valores previstos — {best_model}")
plt.xlabel("Duração prevista do voo (minutos)")
plt.ylabel("Resíduo = real - previsto (minutos)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_residuos_vs_previstos_melhor_modelo.png", dpi=300)
plt.savefig(FIG_DIR / "06_residuos_vs_previstos_melhor_modelo.pdf")
plt.close()

# -----------------------------
# Interpretação automática simples
# -----------------------------
residual_mean = residuals.mean()
residual_skew = residuals.skew()
large_error_threshold = residuals.abs().quantile(0.95)
large_errors = pred_df[pred_df["erro_absoluto"] >= large_error_threshold]

notes = []
notes.append("# Notas da Análise de Resíduos\n")
notes.append(f"- Modelo analisado: `{best_model}`.\n")
notes.append(f"- Média dos resíduos: **{residual_mean:.4f} minutos**. Idealmente deve estar próxima de zero.\n")
notes.append(f"- Skewness dos resíduos: **{residual_skew:.4f}**. Valores muito afastados de zero sugerem assimetria nos erros.\n")
notes.append(f"- Limiar dos 5% maiores erros absolutos: **{large_error_threshold:.4f} minutos**.\n")
notes.append(f"- Número de casos nos 5% maiores erros absolutos: **{len(large_errors)}**.\n")
notes.append("- Um histograma centrado em zero sugere ausência de enviesamento médio forte.\n")
notes.append("- Um padrão em funil no gráfico resíduos vs. previstos sugere heterocedasticidade.\n")
notes.append("- Um padrão curvo sugere não linearidade ainda não capturada pelo modelo.\n")
notes.append("- Outliers podem aumentar muito o RMSE; nesses casos, deve analisar-se se são erros de dados, voos raros ou operações legítimas.\n")
notes.append("- Antes de remover outliers, deve existir uma justificação operacional ou estatística clara.\n")

(NOTE_DIR / "06_notas_residuos.md").write_text("".join(notes), encoding="utf-8")

print("\nFicheiros gerados:")
print(f"- Resíduos CSV: {TAB_DIR / '06_residuos_melhor_modelo.csv'}")
print(f"- Histograma PNG/PDF: {FIG_DIR / '06_histograma_residuos_melhor_modelo.png'}")
print(f"- Dispersão PNG/PDF: {FIG_DIR / '06_residuos_vs_previstos_melhor_modelo.png'}")
print(f"- Notas: {NOTE_DIR / '06_notas_residuos.md'}")
print("\n06 - Análise de resíduos concluída com sucesso.")
