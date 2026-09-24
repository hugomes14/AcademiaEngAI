# 05_grafico_previsto_vs_real.py
# Lab 06.1 - Regressão — Estimar a Duração de Voo
# Objetivo: criar o gráfico Previsto vs. Real para o melhor modelo pelo critério RMSE.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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
print("05 - GRÁFICO PREVISTO VS. REAL")
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
y_real = pred_df["y_real"]
y_previsto = pred_df["y_previsto"]
r2 = r2_score(y_real, y_previsto)
errors = y_previsto - y_real
mean_error = errors.mean()
mae_like = errors.abs().mean()

print(f"Melhor modelo: {best_model}")
print(f"R² no conjunto de teste: {r2:.4f}")
print(f"Erro médio previsto-real: {mean_error:.4f} minutos")
print(f"Erro absoluto médio calculado para apoio interpretativo: {mae_like:.4f} minutos")

min_value = min(y_real.min(), y_previsto.min())
max_value = max(y_real.max(), y_previsto.max())
margin = (max_value - min_value) * 0.05
line_min = min_value - margin
line_max = max_value + margin

plt.figure(figsize=(8, 8))
plt.scatter(y_real, y_previsto, alpha=0.75)
plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", linewidth=2)
plt.title(f"Previsto vs. Real — {best_model}\nR² = {r2:.4f}")
plt.xlabel("Duração real do voo (minutos)")
plt.ylabel("Duração prevista do voo (minutos)")
plt.xlim(line_min, line_max)
plt.ylim(line_min, line_max)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_previsto_vs_real_melhor_modelo.png", dpi=300)
plt.savefig(FIG_DIR / "05_previsto_vs_real_melhor_modelo.pdf")
plt.close()

# -----------------------------
# Análise simples de sub/sobre-estimação
# -----------------------------
under_estimation_count = int((y_previsto < y_real).sum())
over_estimation_count = int((y_previsto > y_real).sum())
near_count = int((errors.abs() <= 5).sum())

notes = []
notes.append("# Notas do Gráfico Previsto vs. Real\n")
notes.append(f"- Melhor modelo pelo RMSE: `{best_model}`.\n")
notes.append(f"- R² no conjunto de teste: **{r2:.4f}**.\n")
notes.append(f"- Número de casos em que o modelo sub-estimou a duração: **{under_estimation_count}**.\n")
notes.append(f"- Número de casos em que o modelo sobre-estimou a duração: **{over_estimation_count}**.\n")
notes.append(f"- Número de previsões até 5 minutos de erro absoluto: **{near_count}**.\n")
notes.append("- Pontos próximos da linha de 45 graus indicam previsões próximas dos valores reais.\n")
notes.append("- Pontos abaixo da linha representam sub-estimação; pontos acima representam sobre-estimação.\n")
notes.append("- Em contexto operacional, a sub-estimação pode ser mais crítica, porque pode reduzir margens de segurança e planeamento.\n")

(NOTE_DIR / "05_notas_previsto_vs_real.md").write_text("".join(notes), encoding="utf-8")

summary_df = pd.DataFrame({
    "metrica": ["melhor_modelo", "R2", "erro_medio_previsto_menos_real", "n_sub_estimacoes", "n_sobre_estimacoes", "n_erros_ate_5_min"],
    "valor": [best_model, r2, mean_error, under_estimation_count, over_estimation_count, near_count]
})
summary_df.to_csv(TAB_DIR / "05_resumo_previsto_vs_real.csv", index=False)

print("\nFicheiros gerados:")
print(f"- PNG: {FIG_DIR / '05_previsto_vs_real_melhor_modelo.png'}")
print(f"- PDF: {FIG_DIR / '05_previsto_vs_real_melhor_modelo.pdf'}")
print(f"- Notas: {NOTE_DIR / '05_notas_previsto_vs_real.md'}")
print("\n05 - Gráfico Previsto vs. Real concluído com sucesso.")
