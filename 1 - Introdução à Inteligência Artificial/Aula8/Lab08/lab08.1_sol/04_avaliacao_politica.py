"""
04 — Avaliação da política aprendida.

Objetivo:
- Avaliar o agente treinado sem exploração.
- Comparar com um agente aleatório.
- Guardar tabela de métricas.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from rl_ambientes import GridWorld

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "artefactos" / "modelos"
TABELAS_DIR = BASE_DIR / "artefactos" / "tabelas"
TABELAS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("LAB 08.1 — AVALIAÇÃO DA POLÍTICA")
print("=" * 80)

q_table_path = MODELOS_DIR / "q_table.npy"
if not q_table_path.exists():
    raise FileNotFoundError("Falta q_table.npy. Corre primeiro o script 03_treino_qlearning.py.")

q_table = np.load(q_table_path)
env = GridWorld(seed=999)

n_eval_episodes = 300
max_steps_per_episode = 200
rng = np.random.default_rng(999)

linhas_metricas = []
detalhe_episodios = []

for agente in ["agente_treinado", "agente_aleatorio"]:
    recompensas = []
    sucessos = []
    passos_lista = []

    for episodio in range(1, n_eval_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        success = 0
        steps_taken = max_steps_per_episode

        for step in range(1, max_steps_per_episode + 1):
            if agente == "agente_treinado":
                best_value = np.max(q_table[state, :])
                best_actions = np.flatnonzero(q_table[state, :] == best_value)
                action = int(rng.choice(best_actions))
            else:
                action = env.sample_action()

            new_state, reward, terminated, info = env.step(action)
            total_reward += reward
            state = new_state
            steps_taken = step

            if terminated:
                success = 1 if info["motivo"] == "destino" else 0
                break

        recompensas.append(total_reward)
        sucessos.append(success)
        passos_lista.append(steps_taken)
        detalhe_episodios.append({
            "agente": agente,
            "episodio": episodio,
            "recompensa_total": total_reward,
            "sucesso": success,
            "passos": steps_taken,
        })

    linhas_metricas.append({
        "agente": agente,
        "episodios": n_eval_episodes,
        "recompensa_media": float(np.mean(recompensas)),
        "recompensa_desvio_padrao": float(np.std(recompensas)),
        "taxa_sucesso": float(np.mean(sucessos)),
        "passos_medios": float(np.mean(passos_lista)),
    })

df_metricas = pd.DataFrame(linhas_metricas)
df_detalhe = pd.DataFrame(detalhe_episodios)

df_metricas.to_csv(TABELAS_DIR / "qlearning_avaliacao_metricas.csv", index=False)
df_metricas.to_markdown(TABELAS_DIR / "qlearning_avaliacao_metricas.md", index=False)
df_detalhe.to_csv(TABELAS_DIR / "qlearning_avaliacao_detalhe.csv", index=False)

print("\nMétricas de avaliação:")
print(df_metricas.to_string(index=False))

print("\nInterpretação:")
print("- A política treinada deve ter maior recompensa média e maior taxa de sucesso do que o agente aleatório.")
print("- A avaliação usa epsilon=0: o agente explora a melhor política aprendida, sem ações aleatórias intencionais.")
print("- Em problemas reais, a política deve ser avaliada em vários episódios para reduzir o efeito da aleatoriedade.")

print("\nFicheiros criados:")
print(f"- {TABELAS_DIR / 'qlearning_avaliacao_metricas.csv'}")
print(f"- {TABELAS_DIR / 'qlearning_avaliacao_metricas.md'}")
print(f"- {TABELAS_DIR / 'qlearning_avaliacao_detalhe.csv'}")
