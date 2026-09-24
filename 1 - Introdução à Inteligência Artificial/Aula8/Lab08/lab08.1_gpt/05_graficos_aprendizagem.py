"""
05 — Gráficos de aprendizagem.

Objetivo:
- Gerar a curva de recompensa média móvel.
- Gerar a curva de taxa de sucesso média móvel.
- Mostrar se o agente melhora ao longo do treino.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
TABELAS_DIR = BASE_DIR / "artefactos" / "tabelas"
IMAGENS_DIR = BASE_DIR / "artefactos" / "imagens"
IMAGENS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("LAB 08.1 — GRÁFICOS DE APRENDIZAGEM")
print("=" * 80)

historico_path = TABELAS_DIR / "qlearning_historico_treino.csv"
if not historico_path.exists():
    raise FileNotFoundError("Falta qlearning_historico_treino.csv. Corre primeiro o script 03_treino_qlearning.py.")

df = pd.read_csv(historico_path)

plt.figure(figsize=(11, 6))
sns.lineplot(data=df, x="episodio", y="recompensa_total", alpha=0.25, label="Recompensa por episódio")
sns.lineplot(data=df, x="episodio", y="recompensa_media_movel_100", label="Média móvel 100 episódios")
plt.title("Q-Learning — Curva de aprendizagem")
plt.xlabel("Episódio")
plt.ylabel("Recompensa")
plt.tight_layout()
plt.savefig(IMAGENS_DIR / "qlearning_curva_aprendizagem.png", dpi=180)
plt.savefig(IMAGENS_DIR / "qlearning_curva_aprendizagem.pdf")
plt.close()

plt.figure(figsize=(11, 6))
sns.lineplot(data=df, x="episodio", y="taxa_sucesso_movel_100")
plt.title("Q-Learning — Taxa de sucesso média móvel")
plt.xlabel("Episódio")
plt.ylabel("Taxa de sucesso nos últimos 100 episódios")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(IMAGENS_DIR / "qlearning_taxa_sucesso.png", dpi=180)
plt.savefig(IMAGENS_DIR / "qlearning_taxa_sucesso.pdf")
plt.close()

plt.figure(figsize=(11, 6))
sns.lineplot(data=df, x="episodio", y="epsilon")
plt.title("Q-Learning — Decaimento de epsilon")
plt.xlabel("Episódio")
plt.ylabel("Epsilon")
plt.tight_layout()
plt.savefig(IMAGENS_DIR / "qlearning_epsilon.png", dpi=180)
plt.savefig(IMAGENS_DIR / "qlearning_epsilon.pdf")
plt.close()

print("Gráficos criados:")
print(f"- {IMAGENS_DIR / 'qlearning_curva_aprendizagem.png'}")
print(f"- {IMAGENS_DIR / 'qlearning_taxa_sucesso.png'}")
print(f"- {IMAGENS_DIR / 'qlearning_epsilon.png'}")

print("\nInterpretação:")
print("- Se a recompensa média móvel sobe, o agente melhora a política.")
print("- Se a taxa de sucesso sobe, o agente chega ao destino com maior frequência.")
print("- O decaimento de epsilon reduz a exploração ao longo do treino.")
