"""
03 — Treino do agente com Q-Learning no GridWorld.

Objetivo:
- Criar uma Q-table.
- Treinar uma política com epsilon-greedy decrescente.
- Guardar Q-table, histórico de recompensas e hiperparâmetros.
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from rl_ambientes import GridWorld

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "artefactos" / "modelos"
TABELAS_DIR = BASE_DIR / "artefactos" / "tabelas"
for pasta in [MODELOS_DIR, TABELAS_DIR]:
    pasta.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("LAB 08.1 — TREINO Q-LEARNING")
print("=" * 80)

env = GridWorld(seed=123)

# Hiperparâmetros principais.
num_episodes = 8000
max_steps_per_episode = 200
learning_rate = 0.12       # alpha: peso dado à nova informação.
discount_factor = 0.95     # gamma: importância de recompensas futuras.
epsilon = 1.0              # começa com muita exploração.
min_epsilon = 0.03
epsilon_decay_rate = 0.9992

q_table = np.zeros((env.n_states, env.n_actions), dtype=float)

rewards_history = []
steps_history = []
success_history = []
epsilon_history = []

rng = np.random.default_rng(123)

print("Descrição do ambiente:")
print(json.dumps(env.describe(), indent=2, ensure_ascii=False))

print("\nHiperparâmetros:")
print(f"- Episódios: {num_episodes}")
print(f"- Máximo de passos por episódio: {max_steps_per_episode}")
print(f"- Alpha/Learning rate: {learning_rate}")
print(f"- Gamma/Discount factor: {discount_factor}")
print(f"- Epsilon inicial: {epsilon}")
print(f"- Epsilon mínimo: {min_epsilon}")
print(f"- Decaimento de epsilon: {epsilon_decay_rate}")

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0.0
    success = 0
    steps_taken = 0

    for step in range(1, max_steps_per_episode + 1):
        # Política epsilon-greedy: explora com probabilidade epsilon.
        if rng.random() < epsilon:
            action = env.sample_action()
        else:
            # Em caso de empate, escolhe aleatoriamente uma das melhores ações.
            best_value = np.max(q_table[state, :])
            best_actions = np.flatnonzero(q_table[state, :] == best_value)
            action = int(rng.choice(best_actions))

        new_state, reward, terminated, info = env.step(action)

        # Equação de Bellman:
        # Q(s,a) <- (1-alpha)Q(s,a) + alpha * (reward + gamma * max Q(s',a'))
        if terminated:
            target = reward
        else:
            target = reward + discount_factor * np.max(q_table[new_state, :])

        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + learning_rate * target

        state = new_state
        total_reward += reward
        steps_taken = step

        if terminated:
            success = 1 if info["motivo"] == "destino" else 0
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)

    rewards_history.append(total_reward)
    steps_history.append(steps_taken)
    success_history.append(success)
    epsilon_history.append(epsilon)

    if episode % 1000 == 0:
        media_ultimos = np.mean(rewards_history[-100:])
        sucesso_ultimos = np.mean(success_history[-100])
        print(
            f"Episódio {episode:05d} | recompensa média últimos 100={media_ultimos:.3f} "
            f"| sucesso últimos 100={sucesso_ultimos:.2%} | epsilon={epsilon:.3f}"
        )

df_historico = pd.DataFrame({
    "episodio": np.arange(1, num_episodes + 1),
    "recompensa_total": rewards_history,
    "passos": steps_history,
    "sucesso": success_history,
    "epsilon": epsilon_history,
})
df_historico["recompensa_media_movel_100"] = df_historico["recompensa_total"].rolling(100).mean()
df_historico["taxa_sucesso_movel_100"] = df_historico["sucesso"].rolling(100).mean()

np.save(MODELOS_DIR / "q_table.npy", q_table)
df_historico.to_csv(TABELAS_DIR / "qlearning_historico_treino.csv", index=False)

with open(MODELOS_DIR / "qlearning_config.pkl", "wb") as f:
    pickle.dump({
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "min_epsilon": min_epsilon,
        "epsilon_decay_rate": epsilon_decay_rate,
        "grid_description": env.describe(),
    }, f)

print("\nTreino concluído.")
print(f"Recompensa média nos últimos 100 episódios: {np.mean(rewards_history[-100:]):.3f}")
print(f"Taxa de sucesso nos últimos 100 episódios: {np.mean(success_history[-100]):.2%}")

print("\nFicheiros criados:")
print(f"- {MODELOS_DIR / 'q_table.npy'}")
print(f"- {TABELAS_DIR / 'qlearning_historico_treino.csv'}")
print(f"- {MODELOS_DIR / 'qlearning_config.pkl'}")
