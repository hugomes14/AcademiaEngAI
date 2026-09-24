"""
01 — Definição e exploração dos ambientes simulados.

Objetivo:
- Explorar um Multi-Armed Bandit com rotas de voo.
- Explorar um Grid World 10x10 para navegação.
- Mostrar estados, ações e recompensas.
"""

from pathlib import Path
import json
import pandas as pd
from rl_ambientes import MultiArmedBandit, GridWorld
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
TABELAS_DIR = BASE_DIR / "artefactos" / "tabelas"
TABELAS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("LAB 08.1 — EXPLORAÇÃO DOS AMBIENTES")
print("=" * 80)

# ---------------------------------------------------------------------
# Ambiente 1: Multi-Armed Bandit
# ---------------------------------------------------------------------
bandit = MultiArmedBandit(
    route_names=["Rota Atlântica", "Rota Interior", "Rota Costeira", "Rota Direta", "Rota Alternativa"],
    means=np.array([1.2, 0.8, 1.5, 1.9, 1.1]),
    stds=np.array([0.35, 0.30, 0.45, 0.55, 0.40]),
    seed=42,
)

print("\n[Bandit] Descrição do ambiente:")
print(json.dumps(bandit.describe(), indent=2, ensure_ascii=False))

print("\n[Bandit] 10 escolhas aleatórias de rota:")
bandit_logs = []
for passo in range(1, 11):
    acao = int(bandit.rng.integers(0, bandit.n_actions))
    recompensa = bandit.step(acao)
    linha = {
        "passo": passo,
        "acao": acao,
        "rota": bandit.route_names[acao],
        "recompensa": recompensa,
    }
    bandit_logs.append(linha)
    print(f"  Passo {passo:02d} | Ação={acao} | {bandit.route_names[acao]} | Recompensa={recompensa:.3f}")

pd.DataFrame(bandit_logs).to_csv(TABELAS_DIR / "exploracao_bandit.csv", index=False)

# ---------------------------------------------------------------------
# Ambiente 2: Grid World
# ---------------------------------------------------------------------
env = GridWorld(seed=42)

print("\n[GridWorld] Descrição do ambiente:")
print(json.dumps(env.describe(), indent=2, ensure_ascii=False))

estado = env.reset()
print(f"\n[GridWorld] Estado inicial: {estado} | posição={env.index_to_state(estado)}")

print("\n[GridWorld] 10 passos aleatórios:")
grid_logs = []
for passo in range(1, 11):
    acao = env.sample_action()
    novo_estado, recompensa, terminado, info = env.step(acao)
    linha = {
        "passo": passo,
        "estado": estado,
        "posicao": str(env.index_to_state(estado)),
        "acao": acao,
        "acao_simbolo": env.ACTION_SYMBOLS[acao],
        "novo_estado": novo_estado,
        "nova_posicao": str(env.index_to_state(novo_estado)),
        "recompensa": recompensa,
        "terminado": terminado,
        "motivo": info["motivo"],
    }
    grid_logs.append(linha)
    print(
        f"  Passo {passo:02d} | estado={estado:02d} -> {novo_estado:02d} "
        f"| ação={env.ACTION_SYMBOLS[acao]} | recompensa={recompensa:.1f} | terminado={terminado}"
    )
    estado = novo_estado
    if terminado:
        print("  Episódio terminou. O ambiente será reiniciado para continuar a demonstração.")
        estado = env.reset()

pd.DataFrame(grid_logs).to_csv(TABELAS_DIR / "exploracao_gridworld.csv", index=False)

print("\nNotas didáticas:")
print("- No Bandit, não existe estado relevante: o problema é escolher a melhor ação ao longo do tempo.")
print("- No GridWorld, cada posição da grelha é um estado e cada movimento é uma ação.")
print("- A recompensa define o comportamento esperado do agente. Uma recompensa mal definida pode ensinar a política errada.")
print("- Em RL, o agente aprende por tentativa e erro, sem precisar de um CSV histórico.")

print("\nFicheiros criados:")
print(f"- {TABELAS_DIR / 'exploracao_bandit.csv'}")
print(f"- {TABELAS_DIR / 'exploracao_gridworld.csv'}")
