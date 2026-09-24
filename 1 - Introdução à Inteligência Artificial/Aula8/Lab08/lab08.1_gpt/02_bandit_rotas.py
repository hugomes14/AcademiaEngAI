"""
02 — Multi-Armed Bandit: Epsilon-Greedy, UCB e Thompson Sampling.

Objetivo:
- Comparar estratégias de exploração vs. explotação.
- Medir recompensa acumulada e regret acumulado.
- Guardar métricas e gráficos.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rl_ambientes import MultiArmedBandit

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "artefactos" / "modelos"
TABELAS_DIR = BASE_DIR / "artefactos" / "tabelas"
IMAGENS_DIR = BASE_DIR / "artefactos" / "imagens"
for pasta in [MODELOS_DIR, TABELAS_DIR, IMAGENS_DIR]:
    pasta.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("LAB 08.1 — MULTI-ARMED BANDIT")
print("=" * 80)

NUM_RUNS = 40
NUM_STEPS = 2500
EPSILON = 0.1
UCB_C = 2.0

route_names = ["Rota Atlântica", "Rota Interior", "Rota Costeira", "Rota Direta", "Rota Alternativa"]
means = np.array([1.2, 0.8, 1.5, 1.9, 1.1])
stds = np.array([0.35, 0.30, 0.45, 0.55, 0.40])
optimal_mean = float(np.max(means))
optimal_action = int(np.argmax(means))

print(f"Rotas disponíveis: {route_names}")
print(f"Melhor rota real: {route_names[optimal_action]} | recompensa média real={optimal_mean:.3f}")
print("Nota: o agente não conhece estas médias. Ele estima através de tentativa e erro.")

algoritmos = ["epsilon_greedy", "ucb", "thompson_sampling"]
resultados_por_algoritmo = {algo: [] for algo in algoritmos}
estimativas_finais = []

for run in range(NUM_RUNS):
    for algoritmo in algoritmos:
        bandit = MultiArmedBandit(route_names, means, stds, seed=42 + run)

        estimativas = np.zeros(bandit.n_actions)
        contagens = np.zeros(bandit.n_actions)
        recompensas = []
        regrets = []

        for passo in range(1, NUM_STEPS + 1):
            if algoritmo == "epsilon_greedy":
                if bandit.rng.random() < EPSILON:
                    acao = int(bandit.rng.integers(0, bandit.n_actions))
                else:
                    acao = int(np.argmax(estimativas))

            elif algoritmo == "ucb":
                # Nos primeiros passos, testa cada rota pelo menos uma vez.
                if np.any(contagens == 0):
                    acao = int(np.argmin(contagens))
                else:
                    bonus = UCB_C * np.sqrt(np.log(passo) / contagens)
                    acao = int(np.argmax(estimativas + bonus))

            else:
                # Thompson Sampling aproximado para recompensas contínuas.
                # Rotas pouco testadas têm maior incerteza, logo têm maior probabilidade de exploração.
                incerteza = 1.0 / np.sqrt(contagens + 1.0)
                amostras = bandit.rng.normal(estimativas, incerteza)
                acao = int(np.argmax(amostras))

            recompensa = bandit.step(acao)

            contagens[acao] += 1
            estimativas[acao] += (recompensa - estimativas[acao]) / contagens[acao]

            recompensas.append(recompensa)
            regrets.append(optimal_mean - recompensa)

        df_run = pd.DataFrame({
            "run": run,
            "passo": np.arange(1, NUM_STEPS + 1),
            "algoritmo": algoritmo,
            "recompensa": recompensas,
            "regret": regrets,
            "recompensa_media_acumulada": np.cumsum(recompensas) / np.arange(1, NUM_STEPS + 1),
            "regret_acumulado": np.cumsum(regrets),
        })
        resultados_por_algoritmo[algoritmo].append(df_run)

        for idx, rota in enumerate(route_names):
            estimativas_finais.append({
                "run": run,
                "algoritmo": algoritmo,
                "rota": rota,
                "estimativa_final": estimativas[idx],
                "selecoes": contagens[idx],
            })

df_resultados = pd.concat(
    [pd.concat(lista, ignore_index=True) for lista in resultados_por_algoritmo.values()],
    ignore_index=True,
)

df_metricas = (
    df_resultados
    .groupby(["algoritmo", "passo"], as_index=False)
    .agg(
        recompensa_media=("recompensa", "mean"),
        recompensa_media_acumulada=("recompensa_media_acumulada", "mean"),
        regret_acumulado=("regret_acumulado", "mean"),
    )
)

df_resumo = (
    df_resultados[df_resultados["passo"] == NUM_STEPS]
    .groupby("algoritmo", as_index=False)
    .agg(
        recompensa_media_final=("recompensa_media_acumulada", "mean"),
        regret_acumulado_final=("regret_acumulado", "mean"),
    )
    .sort_values("regret_acumulado_final")
)

df_estimativas = pd.DataFrame(estimativas_finais)

df_metricas.to_csv(TABELAS_DIR / "bandit_metricas_por_passo.csv", index=False)
df_resumo.to_csv(TABELAS_DIR / "bandit_resumo.csv", index=False)
df_resumo.to_markdown(TABELAS_DIR / "bandit_resumo.md", index=False)
df_estimativas.to_csv(TABELAS_DIR / "bandit_estimativas_finais.csv", index=False)

with open(MODELOS_DIR / "bandit_config.pkl", "wb") as f:
    pickle.dump({
        "route_names": route_names,
        "means": means,
        "stds": stds,
        "num_runs": NUM_RUNS,
        "num_steps": NUM_STEPS,
        "epsilon": EPSILON,
        "ucb_c": UCB_C,
        "optimal_action": optimal_action,
        "optimal_mean": optimal_mean,
    }, f)

print("\nResumo final:")
print(df_resumo.to_string(index=False))

# Gráfico 1: recompensa média acumulada
plt.figure(figsize=(11, 6))
sns.lineplot(data=df_metricas, x="passo", y="recompensa_media_acumulada", hue="algoritmo")
plt.title("Bandit — Recompensa média acumulada por algoritmo")
plt.xlabel("Passo")
plt.ylabel("Recompensa média acumulada")
plt.tight_layout()
plt.savefig(IMAGENS_DIR / "bandit_recompensa_media_acumulada.png", dpi=180)
plt.savefig(IMAGENS_DIR / "bandit_recompensa_media_acumulada.pdf")
plt.close()

# Gráfico 2: regret acumulado
plt.figure(figsize=(11, 6))
sns.lineplot(data=df_metricas, x="passo", y="regret_acumulado", hue="algoritmo")
plt.title("Bandit — Regret acumulado por algoritmo")
plt.xlabel("Passo")
plt.ylabel("Regret acumulado")
plt.tight_layout()
plt.savefig(IMAGENS_DIR / "bandit_regret_acumulado.png", dpi=180)
plt.savefig(IMAGENS_DIR / "bandit_regret_acumulado.pdf")
plt.close()

print("\nFicheiros criados:")
print(f"- {TABELAS_DIR / 'bandit_metricas_por_passo.csv'}")
print(f"- {TABELAS_DIR / 'bandit_resumo.csv'}")
print(f"- {IMAGENS_DIR / 'bandit_recompensa_media_acumulada.png'}")
print(f"- {IMAGENS_DIR / 'bandit_regret_acumulado.png'}")
