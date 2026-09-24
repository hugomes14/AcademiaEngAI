"""
06 — Visualização da política e da função valor.

Objetivo:
- Calcular V(s) = max_a Q(s,a).
- Criar heatmap da função valor.
- Criar mapa da política ótima com setas.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rl_ambientes import GridWorld

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "artefactos" / "modelos"
TABELAS_DIR = BASE_DIR / "artefactos" / "tabelas"
IMAGENS_DIR = BASE_DIR / "artefactos" / "imagens"
for pasta in [TABELAS_DIR, IMAGENS_DIR]:
    pasta.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("LAB 08.1 — VISUALIZAÇÃO DA POLÍTICA")
print("=" * 80)

q_table_path = MODELOS_DIR / "q_table.npy"
if not q_table_path.exists():
    raise FileNotFoundError("Falta q_table.npy. Corre primeiro o script 03_treino_qlearning.py.")

q_table = np.load(q_table_path)
env = GridWorld(seed=123)

valores = np.max(q_table, axis=1).reshape(env.size, env.size)
melhores_acoes = np.argmax(q_table, axis=1).reshape(env.size, env.size)

# Estados terminais com anotação própria.
anotacoes_valor = valores.round(2).astype(str)
anotacoes_politica = np.empty((env.size, env.size), dtype=object)

linhas_politica = []
for row in range(env.size):
    for col in range(env.size):
        pos = (row, col)
        state = env.state_to_index(pos)

        if pos == env.start:
            simbolo = "S"
        elif pos == env.goal:
            simbolo = "G"
        elif pos in env.obstacles:
            simbolo = "X"
        else:
            acao = int(melhores_acoes[row, col])
            simbolo = env.ACTION_SYMBOLS[acao]

        anotacoes_politica[row, col] = simbolo
        linhas_politica.append({
            "linha": row,
            "coluna": col,
            "estado": state,
            "tipo": "inicio" if pos == env.start else "destino" if pos == env.goal else "obstaculo" if pos in env.obstacles else "normal",
            "valor": valores[row, col],
            "melhor_acao": int(melhores_acoes[row, col]),
            "simbolo": simbolo,
        })

df_politica = pd.DataFrame(linhas_politica)
df_politica.to_csv(TABELAS_DIR / "qlearning_politica_grid.csv", index=False)

plt.figure(figsize=(9, 7))
sns.heatmap(valores, annot=anotacoes_valor, fmt="", cmap="viridis", cbar=True)
plt.title("Q-Learning — Função Valor V(s)")
plt.xlabel("Coluna")
plt.ylabel("Linha")
plt.tight_layout()
plt.savefig(IMAGENS_DIR / "qlearning_funcao_valor_heatmap.png", dpi=180)
plt.savefig(IMAGENS_DIR / "qlearning_funcao_valor_heatmap.pdf")
plt.close()

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(0, env.size)
ax.set_ylim(0, env.size)
ax.set_xticks(range(env.size + 1))
ax.set_yticks(range(env.size + 1))
ax.grid(True)
ax.invert_yaxis()
ax.set_title("Q-Learning — Política ótima aprendida")
ax.set_xlabel("Coluna")
ax.set_ylabel("Linha")

for row in range(env.size):
    for col in range(env.size):
        pos = (row, col)
        texto = anotacoes_politica[row, col]

        if pos in env.obstacles:
            ax.add_patch(plt.Rectangle((col, row), 1, 1, alpha=0.35))
        elif pos == env.goal:
            ax.add_patch(plt.Rectangle((col, row), 1, 1, alpha=0.25))
        elif pos == env.start:
            ax.add_patch(plt.Rectangle((col, row), 1, 1, alpha=0.15))

        ax.text(col + 0.5, row + 0.5, texto, ha="center", va="center", fontsize=16)

plt.tight_layout()
plt.savefig(IMAGENS_DIR / "qlearning_politica_setas.png", dpi=180)
plt.savefig(IMAGENS_DIR / "qlearning_politica_setas.pdf")
plt.close()

print("Ficheiros criados:")
print(f"- {TABELAS_DIR / 'qlearning_politica_grid.csv'}")
print(f"- {IMAGENS_DIR / 'qlearning_funcao_valor_heatmap.png'}")
print(f"- {IMAGENS_DIR / 'qlearning_politica_setas.png'}")

print("\nLegenda da política:")
print("- S: início")
print("- G: destino")
print("- X: obstáculo")
print("- ↑ ↓ ← →: melhor ação estimada pela Q-table")
