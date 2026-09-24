# 04_avaliacao_metricas.py
# Lab 06.1 - Regressão — Estimar a Duração de Voo
# Objetivo: calcular R², MAE, MSE e RMSE para todos os modelos treinados.

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# Configuração de caminhos
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
PRED_DIR = OUTPUT_DIR / "previsoes"
TAB_DIR = OUTPUT_DIR / "tabelas"
NOTE_DIR = OUTPUT_DIR / "notas"

TAB_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("04 - AVALIAÇÃO E MÉTRICAS")
print("=" * 80)

prediction_files = sorted(PRED_DIR.glob("previsoes_*.csv"))
prediction_files = [path for path in prediction_files if path.name != "previsoes_todos_modelos.csv"]

if not prediction_files:
    raise FileNotFoundError(f"Não foram encontradas previsões em {PRED_DIR}. Corre primeiro o script 03.")

metric_rows = []

# -----------------------------
# Cálculo de métricas por modelo
# -----------------------------
for pred_file in prediction_files:
    pred_df = pd.read_csv(pred_file)
    required_columns = ["y_real", "y_previsto"]
    missing_columns = [col for col in required_columns if col not in pred_df.columns]
    if missing_columns:
        raise ValueError(f"Ficheiro {pred_file.name} sem colunas obrigatórias: {missing_columns}")

    y_real = pred_df["y_real"]
    y_previsto = pred_df["y_previsto"]

    r2 = r2_score(y_real, y_previsto)
    mae = mean_absolute_error(y_real, y_previsto)
    mse = mean_squared_error(y_real, y_previsto)
    rmse = np.sqrt(mse)

    model_name = pred_file.stem.replace("previsoes_", "")
    metric_rows.append({
        "modelo": model_name,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "ficheiro_previsoes": pred_file.name
    })

metrics_df = pd.DataFrame(metric_rows)
metrics_df = metrics_df.sort_values(by=["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
metrics_df["ranking_RMSE"] = metrics_df["RMSE"].rank(method="min", ascending=True).astype(int)
metrics_df["melhor_RMSE"] = metrics_df["RMSE"] == metrics_df["RMSE"].min()
metrics_df["melhor_MAE"] = metrics_df["MAE"] == metrics_df["MAE"].min()
metrics_df["melhor_R2"] = metrics_df["R2"] == metrics_df["R2"].max()

print("\nTabela de métricas ordenada por RMSE:")
print(metrics_df[["modelo", "R2", "MAE", "MSE", "RMSE", "ranking_RMSE"]].round(4))

metrics_df.to_csv(TAB_DIR / "04_metricas_modelos.csv", index=False)
metrics_df.round(4).to_csv(TAB_DIR / "04_metricas_modelos_formatado.csv", index=False)

# -----------------------------
# Markdown com destaque visual
# -----------------------------
markdown_rows = []
markdown_rows.append("| Ranking RMSE | Modelo | R² | MAE | MSE | RMSE |")
markdown_rows.append("|---:|---|---:|---:|---:|---:|")
for _, row in metrics_df.iterrows():
    model_display = row["modelo"]
    rmse_display = f"{row['RMSE']:.4f}"
    mae_display = f"{row['MAE']:.4f}"
    r2_display = f"{row['R2']:.4f}"
    mse_display = f"{row['MSE']:.4f}"
    if row["melhor_RMSE"]:
        model_display = f"**{model_display}**"
        rmse_display = f"**{rmse_display}**"
    if row["melhor_MAE"]:
        mae_display = f"**{mae_display}**"
    if row["melhor_R2"]:
        r2_display = f"**{r2_display}**"
    markdown_rows.append(
        f"| {int(row['ranking_RMSE'])} | {model_display} | {r2_display} | {mae_display} | {mse_display} | {rmse_display} |"
    )

markdown_text = "\n".join(markdown_rows) + "\n"
(TAB_DIR / "04_metricas_modelos.md").write_text(markdown_text, encoding="utf-8")

best_model = metrics_df.iloc[0]["modelo"]
print(f"\nMelhor modelo pelo critério RMSE: {best_model}")
(TAB_DIR / "04_melhor_modelo.txt").write_text(str(best_model), encoding="utf-8")

notes = []
notes.append("# Notas da Avaliação\n")
notes.append(f"- O melhor modelo pelo critério RMSE foi `{best_model}`.\n")
notes.append("- O R² mede a proporção de variância explicada pelo modelo; quanto maior, melhor.\n")
notes.append("- O MAE mede o erro absoluto médio em minutos; é fácil de interpretar e mais robusto a outliers.\n")
notes.append("- O MSE mede o erro quadrático médio; penaliza erros grandes.\n")
notes.append("- O RMSE é a raiz do MSE e fica na unidade original da variável-alvo, minutos; também penaliza erros grandes.\n")
notes.append("- Quando existem outliers ou erros muito grandes, o RMSE tende a aumentar mais do que o MAE.\n")
notes.append("- Em planeamento de voos, sub-estimar a duração pode ser mais problemático do que sobre-estimar, porque pode afetar combustível, janelas operacionais e alocação de recursos.\n")

(NOTE_DIR / "04_notas_avaliacao.md").write_text("".join(notes), encoding="utf-8")

print("\nFicheiros gerados:")
print(f"- CSV: {TAB_DIR / '04_metricas_modelos.csv'}")
print(f"- Markdown: {TAB_DIR / '04_metricas_modelos.md'}")
print(f"- Melhor modelo: {TAB_DIR / '04_melhor_modelo.txt'}")
print("\n04 - Avaliação concluída com sucesso.")
