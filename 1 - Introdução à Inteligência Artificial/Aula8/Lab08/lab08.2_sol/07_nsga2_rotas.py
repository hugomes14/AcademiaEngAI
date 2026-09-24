# 07_nsga2_rotas.py
# Lab 08.2 — Algoritmos Genéticos
# Etapa 7: otimização multiobjetivo de rotas com uma implementação didática de NSGA-II.

from pathlib import Path
import json
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
ROTA_DIR = BASE_DIR / "artefactos" / "nsga2_rotas"
ROTA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ETAPA 7 — Otimização multiobjetivo de rotas")
print("=" * 80)

# Waypoints sintéticos.
# Cada waypoint tem coordenadas e uma zona de vento/risco que influencia consumo.
waypoints = pd.DataFrame({
    "id": list(range(12)),
    "x": [0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 4, 10],
    "y": [0, 3, 1, 4, 2, 7, 8, 7, 8, 6, 5, 10],
    "vento": [0.0, 0.2, -0.1, 0.3, -0.2, 0.5, 0.1, 0.4, -0.1, 0.2, 0.35, 0.0],
})
waypoints.to_csv(ROTA_DIR / "waypoints.csv", index=False)

START_ID = 0
END_ID = 11
INTERMEDIATE_IDS = [i for i in waypoints["id"] if i not in [START_ID, END_ID]]
ROUTE_SIZE = 5

POP_SIZE = 32
NGEN = 12
CXPB = 0.75
MUTPB = 0.25

def distancia(a, b):
    pa = waypoints.loc[waypoints["id"] == a, ["x", "y"]].iloc[0].to_numpy()
    pb = waypoints.loc[waypoints["id"] == b, ["x", "y"]].iloc[0].to_numpy()
    return float(np.linalg.norm(pa - pb))

def avaliar_rota(individual):
    rota = [START_ID] + list(individual) + [END_ID]
    distancia_total = 0.0
    vento_total = 0.0

    for a, b in zip(rota[:-1], rota[1:]):
        d = distancia(a, b)
        distancia_total += d
        vento_total += abs(float(waypoints.loc[waypoints["id"] == b, "vento"].iloc[0])) * d

    # Objetivo 1: minimizar tempo.
    tempo = distancia_total * 5.2 + 0.8 * len(individual)

    # Objetivo 2: minimizar consumo.
    consumo = distancia_total * 3.7 + vento_total * 4.5 + 1.5 * len(set(individual))

    return tempo, consumo

def criar_rota():
    return random.sample(INTERMEDIATE_IDS, ROUTE_SIZE)

def dominates(obj_a, obj_b):
    return (obj_a[0] <= obj_b[0] and obj_a[1] <= obj_b[1]) and (obj_a[0] < obj_b[0] or obj_a[1] < obj_b[1])

def non_dominated_sort(population, objectives):
    S = [[] for _ in population]
    n = [0 for _ in population]
    fronts = [[]]

    for p in range(len(population)):
        for q in range(len(population)):
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]

def crowding_distance(front, objectives):
    distance = {idx: 0.0 for idx in front}

    if len(front) <= 2:
        for idx in front:
            distance[idx] = float("inf")
        return distance

    for m in [0, 1]:
        sorted_front = sorted(front, key=lambda idx: objectives[idx][m])
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")

        min_val = objectives[sorted_front[0]][m]
        max_val = objectives[sorted_front[-1]][m]
        denom = max(max_val - min_val, 1e-9)

        for i in range(1, len(sorted_front) - 1):
            prev_val = objectives[sorted_front[i - 1]][m]
            next_val = objectives[sorted_front[i + 1]][m]
            distance[sorted_front[i]] += (next_val - prev_val) / denom

    return distance

def crossover(p1, p2):
    if random.random() > CXPB:
        return p1[:], p2[:]

    cut = random.randint(1, ROUTE_SIZE - 1)
    c1 = p1[:cut] + [gene for gene in p2 if gene not in p1[:cut]]
    c2 = p2[:cut] + [gene for gene in p1 if gene not in p2[:cut]]
    return c1[:ROUTE_SIZE], c2[:ROUTE_SIZE]

def mutacao(individual):
    child = individual[:]
    if random.random() < MUTPB:
        i, j = random.sample(range(ROUTE_SIZE), 2)
        child[i], child[j] = child[j], child[i]

    if random.random() < MUTPB:
        pos = random.randrange(ROUTE_SIZE)
        available = [w for w in INTERMEDIATE_IDS if w not in child]
        if available:
            child[pos] = random.choice(available)

    return child

def selecionar_nsga2(population, objectives, target_size):
    fronts = non_dominated_sort(population, objectives)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= target_size:
            selected.extend(front)
        else:
            distances = crowding_distance(front, objectives)
            ordered = sorted(front, key=lambda idx: distances[idx], reverse=True)
            selected.extend(ordered[: target_size - len(selected)])
            break

    return [population[idx] for idx in selected]

population = [criar_rota() for _ in range(POP_SIZE)]
log_rows = []

for gen in range(NGEN + 1):
    objectives = [avaliar_rota(ind) for ind in population]
    fronts = non_dominated_sort(population, objectives)
    pareto_size = len(fronts[0])

    best_tempo = min(obj[0] for obj in objectives)
    best_consumo = min(obj[1] for obj in objectives)

    log_rows.append({
        "geracao": gen,
        "pareto_size": pareto_size,
        "melhor_tempo": best_tempo,
        "melhor_consumo": best_consumo,
        "tempo_medio": float(np.mean([obj[0] for obj in objectives])),
        "consumo_medio": float(np.mean([obj[1] for obj in objectives])),
    })

    if gen % 5 == 0:
        print(
            f"Geração {gen:02d} | Pareto={pareto_size} | "
            f"melhor tempo={best_tempo:.2f} | melhor consumo={best_consumo:.2f}"
        )

    if gen == NGEN:
        break

    offspring = []
    while len(offspring) < POP_SIZE:
        p1, p2 = random.sample(population, 2)
        c1, c2 = crossover(p1, p2)
        offspring.append(mutacao(c1))
        if len(offspring) < POP_SIZE:
            offspring.append(mutacao(c2))

    combined = population + offspring
    combined_objectives = [avaliar_rota(ind) for ind in combined]
    population = selecionar_nsga2(combined, combined_objectives, POP_SIZE)

final_objectives = [avaliar_rota(ind) for ind in population]
fronts = non_dominated_sort(population, final_objectives)
pareto_indices = fronts[0]

pareto_rows = []
for idx in pareto_indices:
    individual = population[idx]
    tempo, consumo = final_objectives[idx]
    pareto_rows.append({
        "rota": " -> ".join(map(str, [START_ID] + individual + [END_ID])),
        "waypoints_intermedios": json.dumps(individual),
        "tempo_voo": tempo,
        "consumo_combustivel": consumo,
    })

pareto_df = pd.DataFrame(pareto_rows).sort_values(["tempo_voo", "consumo_combustivel"])
pareto_df.to_csv(ROTA_DIR / "frente_pareto_rotas.csv", index=False)
pareto_df.to_markdown(ROTA_DIR / "frente_pareto_rotas.md", index=False)

log_df = pd.DataFrame(log_rows)
log_df.to_csv(ROTA_DIR / "logbook_nsga2.csv", index=False)

all_df = pd.DataFrame({
    "tempo_voo": [obj[0] for obj in final_objectives],
    "consumo_combustivel": [obj[1] for obj in final_objectives],
})
plt.figure(figsize=(8, 6))
plt.scatter(all_df["tempo_voo"], all_df["consumo_combustivel"], alpha=0.35, label="População final")
plt.scatter(pareto_df["tempo_voo"], pareto_df["consumo_combustivel"], marker="x", s=80, label="Frente de Pareto")
plt.xlabel("Tempo de voo")
plt.ylabel("Consumo de combustível")
plt.title("Frente de Pareto — Rotas de Voo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROTA_DIR / "frente_pareto_rotas.png", dpi=170)
plt.savefig(ROTA_DIR / "frente_pareto_rotas.pdf")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(log_df["geracao"], log_df["melhor_tempo"], label="Melhor tempo")
plt.plot(log_df["geracao"], log_df["melhor_consumo"], label="Melhor consumo")
plt.xlabel("Geração")
plt.ylabel("Valor do objetivo")
plt.title("Evolução dos melhores objetivos")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROTA_DIR / "evolucao_objetivos_nsga2.png", dpi=170)
plt.savefig(ROTA_DIR / "evolucao_objetivos_nsga2.pdf")
plt.close()

md = f"""# Otimização Multiobjetivo de Rotas

## Frente de Pareto

{pareto_df.head(12).to_markdown(index=False)}

## Interpretação

- Cada ponto da Frente de Pareto representa uma rota não dominada.
- Uma rota não dominada não pode melhorar o tempo sem piorar o consumo, ou vice-versa.
- O gráfico mostra o compromisso entre rapidez e eficiência energética.
"""
(ROTA_DIR / "resumo_nsga2_rotas.md").write_text(md, encoding="utf-8")

print("\nNúmero de rotas na Frente de Pareto:", len(pareto_df))
print("Artefactos guardados em:", ROTA_DIR)
print("Etapa 7 concluída com sucesso.")
