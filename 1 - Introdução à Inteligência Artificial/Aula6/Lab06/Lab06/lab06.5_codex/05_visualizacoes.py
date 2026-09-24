# -*- coding: utf-8 -*-
"""
Lab 06.5 - Visualizações finais.

Gráficos produzidos:
1. Curva de aprendizagem PyTorch;
2. Valores reais vs. previstos;
3. Comparação de coeficientes PyTorch vs. sklearn;
4. Distribuição dos resíduos;
5. Resíduos vs. previstos.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

RAIZ = Path(__file__).resolve().parent
DIR_PREV = RAIZ / "artefactos" / "previsoes"
DIR_METRICAS = RAIZ / "artefactos" / "metricas"
DIR_IMG = RAIZ / "imagens"
DIR_IMG.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.5 - VISUALIZAÇÕES")
print("=" * 80)

historico = pd.read_csv(DIR_METRICAS / "historico_treino_pytorch.csv")
prev_pytorch = pd.read_csv(DIR_PREV / "previsoes_pytorch_teste.csv")
prev_sklearn = pd.read_csv(DIR_PREV / "previsoes_sklearn_teste.csv")
coef_comp = pd.read_csv(DIR_METRICAS / "comparacao_coeficientes.csv")

# -----------------------------------------------------------------------------
# 1) Curva de aprendizagem
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(historico["epoca"], historico["loss_treino"], label="Treino")
plt.plot(historico["epoca"], historico["loss_validacao"], label="Validação")
plt.title("Curva de aprendizagem do neurónio linear em PyTorch")
plt.xlabel("Época")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(DIR_IMG / "05_curva_aprendizagem_pytorch.png", dpi=170)
plt.savefig(DIR_IMG / "05_curva_aprendizagem_pytorch.pdf")
plt.close()

# -----------------------------------------------------------------------------
# 2) Valores reais vs. previstos
# -----------------------------------------------------------------------------
plt.figure(figsize=(7, 7))
y_real = prev_pytorch["y_real"]
y_pred = prev_pytorch["y_pred_pytorch"]
lim_min = min(y_real.min(), y_pred.min())
lim_max = max(y_real.max(), y_pred.max())
r2 = r2_score(y_real, y_pred)
plt.scatter(y_real, y_pred, alpha=0.75, label=f"PyTorch | R²={r2:.4f}")
plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", label="Linha ideal")
plt.title("Valores reais vs. previstos - PyTorch")
plt.xlabel("Duração real (min)")
plt.ylabel("Duração prevista (min)")
plt.legend()
plt.tight_layout()
plt.savefig(DIR_IMG / "05_reais_vs_previstos_pytorch.png", dpi=170)
plt.savefig(DIR_IMG / "05_reais_vs_previstos_pytorch.pdf")
plt.close()

plt.figure(figsize=(7, 7))
y_real_sk = prev_sklearn["y_real"]
y_pred_sk = prev_sklearn["y_pred_sklearn"]
lim_min = min(y_real_sk.min(), y_pred_sk.min())
lim_max = max(y_real_sk.max(), y_pred_sk.max())
r2_sk = r2_score(y_real_sk, y_pred_sk)
plt.scatter(y_real_sk, y_pred_sk, alpha=0.75, label=f"sklearn | R²={r2_sk:.4f}")
plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", label="Linha ideal")
plt.title("Valores reais vs. previstos - sklearn")
plt.xlabel("Duração real (min)")
plt.ylabel("Duração prevista (min)")
plt.legend()
plt.tight_layout()
plt.savefig(DIR_IMG / "05_reais_vs_previstos_sklearn.png", dpi=170)
plt.savefig(DIR_IMG / "05_reais_vs_previstos_sklearn.pdf")
plt.close()

# -----------------------------------------------------------------------------
# 3) Comparação de coeficientes
# -----------------------------------------------------------------------------
coef_plot = coef_comp[coef_comp["feature"] != "bias/intercept"].copy()
coef_long = coef_plot.melt(
    id_vars="feature",
    value_vars=["peso_pytorch", "peso_sklearn"],
    var_name="modelo",
    value_name="coeficiente",
)
coef_long["modelo"] = coef_long["modelo"].replace(
    {"peso_pytorch": "PyTorch", "peso_sklearn": "sklearn"}
)

plt.figure(figsize=(11, 6))
sns.barplot(data=coef_long, x="coeficiente", y="feature", hue="modelo")
plt.title("Comparação de coeficientes: PyTorch vs. sklearn")
plt.xlabel("Coeficiente no espaço processado/escalado")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(DIR_IMG / "05_comparacao_coeficientes.png", dpi=170)
plt.savefig(DIR_IMG / "05_comparacao_coeficientes.pdf")
plt.close()

# -----------------------------------------------------------------------------
# 4) Distribuição dos resíduos e resíduos vs. previstos
# -----------------------------------------------------------------------------
residuos = prev_pytorch["y_real"] - prev_pytorch["y_pred_pytorch"]
plt.figure(figsize=(9, 5))
plt.hist(residuos, bins=25, edgecolor="black", alpha=0.85)
plt.axvline(0, linestyle="--")
plt.title("Distribuição dos resíduos - PyTorch")
plt.xlabel("Resíduo = real - previsto (min)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(DIR_IMG / "05_distribuicao_residuos_pytorch.png", dpi=170)
plt.savefig(DIR_IMG / "05_distribuicao_residuos_pytorch.pdf")
plt.close()

plt.figure(figsize=(9, 5))
plt.scatter(prev_pytorch["y_pred_pytorch"], residuos, alpha=0.75)
plt.axhline(0, linestyle="--")
plt.title("Resíduos vs. valores previstos - PyTorch")
plt.xlabel("Valor previsto (min)")
plt.ylabel("Resíduo (min)")
plt.tight_layout()
plt.savefig(DIR_IMG / "05_residuos_vs_previstos_pytorch.png", dpi=170)
plt.savefig(DIR_IMG / "05_residuos_vs_previstos_pytorch.pdf")
plt.close()

print("Gráficos guardados na pasta imagens/.")
