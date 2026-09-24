# 04_grafico_convergencia_ga.py
# Lab 08.2 — Algoritmos Genéticos
# Etapa 4: visualização da convergência do GA.

from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
GA_DIR = BASE_DIR / "artefactos" / "ga_features"
GA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ETAPA 4 — Gráfico de convergência do GA")
print("=" * 80)

logbook_path = GA_DIR / "logbook.pkl"
if not logbook_path.exists():
    raise FileNotFoundError("logbook.pkl não encontrado. Corre primeiro o script 03_ga_selecao_features.py.")

with open(logbook_path, "rb") as f:
    log_rows = pickle.load(f)

logbook = pd.DataFrame(log_rows)
logbook.to_csv(GA_DIR / "logbook_convergencia.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(logbook["geracao"], logbook["fitness_media"], marker="o", label="Fitness média")
plt.plot(logbook["geracao"], logbook["fitness_maxima"], marker="o", label="Fitness máxima")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Convergência do Algoritmo Genético")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(GA_DIR / "convergencia_ga.png", dpi=170)
plt.savefig(GA_DIR / "convergencia_ga.pdf")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(logbook["geracao"], logbook["features_media"], marker="o", label="N.º médio de features")
plt.plot(logbook["geracao"], logbook["features_melhor"], marker="o", label="N.º de features do melhor indivíduo")
plt.xlabel("Geração")
plt.ylabel("Número de features")
plt.title("Tamanho dos subconjuntos ao longo da evolução")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(GA_DIR / "tamanho_subconjuntos_ga.png", dpi=170)
plt.savefig(GA_DIR / "tamanho_subconjuntos_ga.pdf")
plt.close()

md = f"""# Convergência do GA

## Última geração

{logbook.tail(1).to_markdown(index=False)}

## Interpretação

- A linha de fitness máxima mostra o melhor indivíduo encontrado em cada geração.
- A linha de fitness média mostra a qualidade geral da população.
- Se ambas estabilizam, há sinais de convergência.
- Se a fitness máxima sobe mas a média não acompanha, pode existir pouca consistência na população.
"""
(GA_DIR / "analise_convergencia.md").write_text(md, encoding="utf-8")

print("Gráficos guardados:")
print("-", GA_DIR / "convergencia_ga.png")
print("-", GA_DIR / "tamanho_subconjuntos_ga.png")
print("Etapa 4 concluída com sucesso.")
