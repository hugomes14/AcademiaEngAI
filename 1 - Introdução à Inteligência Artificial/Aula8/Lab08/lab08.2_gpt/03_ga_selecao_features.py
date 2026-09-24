# 03_ga_selecao_features.py
# Lab 08.2 — Algoritmos Genéticos
# Etapa 3: Algoritmo Genético para seleção de features.
#
# Nota didática:
# A única função essencial neste script é a função de fitness, porque ela traduz
# o objetivo do AG: escolher features que maximizem o desempenho do modelo.

from pathlib import Path
import json
import pickle
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
GA_DIR = BASE_DIR / "artefactos" / "ga_features"
GA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ETAPA 3 — Algoritmo Genético para seleção de features")
print("=" * 80)

X_train = pd.read_csv(PREP_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(PREP_DIR / "y_train.csv").iloc[:, 0].astype(int)

with open(PREP_DIR / "feature_names.json", "r", encoding="utf-8") as f:
    feature_names = json.load(f)

n_features = X_train.shape[1]
print(f"Número de features disponíveis: {n_features}")

POP_SIZE = 18
NGEN = 8
CXPB = 0.75
MUTPB = 0.08
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3

print("\nHiperparâmetros do AG:")
print(f"POP_SIZE = {POP_SIZE}")
print(f"NGEN = {NGEN}")
print(f"CXPB = {CXPB}")
print(f"MUTPB = {MUTPB}")
print(f"ELITE_SIZE = {ELITE_SIZE}")

fitness_cache = {}

def avaliar_features(individual):
    """
    Função de fitness do Algoritmo Genético.

    Recebe um indivíduo binário:
    - 1 significa que a feature é selecionada
    - 0 significa que a feature é excluída

    A fitness é o F1-score médio em validação cruzada.
    Também existe uma pequena penalização por complexidade para favorecer subconjuntos mais curtos.
    """
    key = tuple(individual)
    if key in fitness_cache:
        return fitness_cache[key]

    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]

    if len(selected_idx) == 0:
        fitness_cache[key] = 0.0
        return 0.0

    X_selected = X_train.iloc[:, selected_idx]

    model = LogisticRegression(
        max_iter=700,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    )

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X_selected, y_train, cv=cv, scoring="f1")

    mean_f1 = float(np.mean(scores))
    complexity_penalty = 0.015 * (len(selected_idx) / n_features)

    fitness = max(mean_f1 - complexity_penalty, 0.0)
    fitness_cache[key] = fitness
    return fitness

def criar_individuo():
    individual = [1 if random.random() < 0.45 else 0 for _ in range(n_features)]
    if sum(individual) == 0:
        individual[random.randrange(n_features)] = 1
    return individual

def inicializar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]

def crossover(parent1, parent2):
    if random.random() > CXPB:
        return parent1[:], parent2[:]

    ponto1 = random.randint(1, n_features - 2)
    ponto2 = random.randint(ponto1 + 1, n_features - 1)

    child1 = parent1[:ponto1] + parent2[ponto1:ponto2] + parent1[ponto2:]
    child2 = parent2[:ponto1] + parent1[ponto1:ponto2] + parent2[ponto2:]
    return child1, child2

def mutacao(individual):
    mutated = individual[:]
    for i in range(n_features):
        if random.random() < MUTPB:
            mutated[i] = 1 - mutated[i]

    if sum(mutated) == 0:
        mutated[random.randrange(n_features)] = 1

    return mutated

def torneio(population, fitnesses):
    candidatos = random.sample(range(len(population)), TOURNAMENT_SIZE)
    melhor_idx = max(candidatos, key=lambda idx: fitnesses[idx])
    return population[melhor_idx][:]

population = inicializar_populacao()
log_rows = []

print("\nA iniciar evolução...")
for gen in range(NGEN + 1):
    fitnesses = [avaliar_features(ind) for ind in population]
    selected_counts = [sum(ind) for ind in population]

    best_idx = int(np.argmax(fitnesses))
    best_ind = population[best_idx]

    row = {
        "geracao": gen,
        "fitness_media": float(np.mean(fitnesses)),
        "fitness_maxima": float(np.max(fitnesses)),
        "fitness_minima": float(np.min(fitnesses)),
        "fitness_std": float(np.std(fitnesses)),
        "features_media": float(np.mean(selected_counts)),
        "features_melhor": int(sum(best_ind)),
    }
    log_rows.append(row)

    print(
        f"Geração {gen:02d} | "
        f"fitness média={row['fitness_media']:.4f} | "
        f"fitness máx={row['fitness_maxima']:.4f} | "
        f"features melhor={row['features_melhor']}"
    )

    if gen == NGEN:
        break

    ranked = sorted(zip(population, fitnesses), key=lambda pair: pair[1], reverse=True)
    elites = [ind[:] for ind, fit in ranked[:ELITE_SIZE]]

    next_population = elites[:]

    while len(next_population) < POP_SIZE:
        p1 = torneio(population, fitnesses)
        p2 = torneio(population, fitnesses)
        c1, c2 = crossover(p1, p2)
        c1 = mutacao(c1)
        c2 = mutacao(c2)
        next_population.extend([c1, c2])

    population = next_population[:POP_SIZE]

final_fitnesses = [avaliar_features(ind) for ind in population]
best_idx = int(np.argmax(final_fitnesses))
best_individual = population[best_idx]
best_fitness = float(final_fitnesses[best_idx])

selected_features = [feature_names[i] for i, bit in enumerate(best_individual) if bit == 1]

logbook = pd.DataFrame(log_rows)
logbook.to_csv(GA_DIR / "logbook.csv", index=False)

with open(GA_DIR / "hof.pkl", "wb") as f:
    pickle.dump([best_individual], f)

with open(GA_DIR / "logbook.pkl", "wb") as f:
    pickle.dump(log_rows, f)

with open(GA_DIR / "population_final.pkl", "wb") as f:
    pickle.dump(population, f)

best_data = {
    "fitness_cv_penalizada": best_fitness,
    "n_features_total": n_features,
    "n_features_selected": len(selected_features),
    "features_selected": selected_features,
    "individual": best_individual,
}
with open(GA_DIR / "best_features_preliminar.json", "w", encoding="utf-8") as f:
    json.dump(best_data, f, ensure_ascii=False, indent=2)

md = f"""# Resultado Preliminar do GA de Seleção de Features

- Fitness CV penalizada: {best_fitness:.4f}
- Features totais: {n_features}
- Features selecionadas: {len(selected_features)}

## Features selecionadas

{chr(10).join(f"- `{feat}`" for feat in selected_features)}

## Nota

A fitness usa F1-score médio em validação cruzada com uma pequena penalização por complexidade.
A avaliação final no conjunto de teste fica para o script `05_avaliacao_melhor_subconjunto.py`.
"""
(GA_DIR / "resultado_ga_features.md").write_text(md, encoding="utf-8")

print("\nMelhor fitness penalizada:", round(best_fitness, 4))
print("Features selecionadas:")
for feat in selected_features:
    print("-", feat)

print("\nArtefactos guardados em:", GA_DIR)
print("Etapa 3 concluída com sucesso.")
