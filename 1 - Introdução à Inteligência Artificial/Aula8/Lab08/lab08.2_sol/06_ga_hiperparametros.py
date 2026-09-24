# 06_ga_hiperparametros.py
# Lab 08.2 — Algoritmos Genéticos
# Etapa 6: otimização simples de hiperparâmetros com AG.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
AVAL_DIR = BASE_DIR / "artefactos" / "avaliacao"
HP_DIR = BASE_DIR / "artefactos" / "ga_hiperparametros"
HP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ETAPA 6 — GA para otimização de hiperparâmetros")
print("=" * 80)

X_train = pd.read_csv(PREP_DIR / "X_train_scaled.csv")
X_test = pd.read_csv(PREP_DIR / "X_test_scaled.csv")
y_train = pd.read_csv(PREP_DIR / "y_train.csv").iloc[:, 0].astype(int)
y_test = pd.read_csv(PREP_DIR / "y_test.csv").iloc[:, 0].astype(int)

best_features_path = AVAL_DIR / "best_features.json"
selected_indices = None
if best_features_path.exists():
    with open(best_features_path, "r", encoding="utf-8") as f:
        best_features = json.load(f)
    selected_indices = best_features.get("selected_indices", None)

if selected_indices:
    print("A usar apenas as features selecionadas pelo GA de features.")
    X_train_fit = X_train.iloc[:, selected_indices]
    X_test_fit = X_test.iloc[:, selected_indices]
else:
    print("best_features.json não encontrado. A usar todas as features.")
    X_train_fit = X_train
    X_test_fit = X_test

# Cromossoma: [n_estimators_idx, max_depth_idx, min_samples_split_idx, max_features_idx]
n_estimators_space = [30, 60, 90]
max_depth_space = [3, 6, None]
min_samples_split_space = [2, 6, 10]
max_features_space = ["sqrt", 0.7, None]

POP_SIZE = 10
NGEN = 4
CXPB = 0.7
MUTPB = 0.18
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3

fitness_cache = {}

def decode(individual):
    return {
        "n_estimators": n_estimators_space[individual[0]],
        "max_depth": max_depth_space[individual[1]],
        "min_samples_split": min_samples_split_space[individual[2]],
        "max_features": max_features_space[individual[3]],
    }

def avaliar_hiperparametros(individual):
    key = tuple(individual)
    if key in fitness_cache:
        return fitness_cache[key]

    params = decode(individual)
    model = RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X_train_fit, y_train, cv=cv, scoring="f1")
    fitness = float(np.mean(scores))
    fitness_cache[key] = fitness
    return fitness

def criar_individuo():
    return [
        random.randrange(len(n_estimators_space)),
        random.randrange(len(max_depth_space)),
        random.randrange(len(min_samples_split_space)),
        random.randrange(len(max_features_space)),
    ]

def crossover(p1, p2):
    if random.random() > CXPB:
        return p1[:], p2[:]
    point = random.randint(1, len(p1)-1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutacao(individual):
    child = individual[:]
    for i in range(len(child)):
        if random.random() < MUTPB:
            if i == 0:
                child[i] = random.randrange(len(n_estimators_space))
            elif i == 1:
                child[i] = random.randrange(len(max_depth_space))
            elif i == 2:
                child[i] = random.randrange(len(min_samples_split_space))
            else:
                child[i] = random.randrange(len(max_features_space))
    return child

def torneio(population, fitnesses):
    candidatos = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_idx = max(candidatos, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]

population = [criar_individuo() for _ in range(POP_SIZE)]
log_rows = []

for gen in range(NGEN + 1):
    fitnesses = [avaliar_hiperparametros(ind) for ind in population]
    best_idx = int(np.argmax(fitnesses))
    best_ind = population[best_idx]
    row = {
        "geracao": gen,
        "fitness_media": float(np.mean(fitnesses)),
        "fitness_maxima": float(np.max(fitnesses)),
        "fitness_minima": float(np.min(fitnesses)),
        "fitness_std": float(np.std(fitnesses)),
        "melhores_parametros": json.dumps(decode(best_ind), ensure_ascii=False),
    }
    log_rows.append(row)

    print(
        f"Geração {gen:02d} | "
        f"fitness média={row['fitness_media']:.4f} | "
        f"fitness máx={row['fitness_maxima']:.4f} | "
        f"params={decode(best_ind)}"
    )

    if gen == NGEN:
        break

    ranked = sorted(zip(population, fitnesses), key=lambda pair: pair[1], reverse=True)
    next_population = [ind[:] for ind, fit in ranked[:ELITE_SIZE]]

    while len(next_population) < POP_SIZE:
        p1 = torneio(population, fitnesses)
        p2 = torneio(population, fitnesses)
        c1, c2 = crossover(p1, p2)
        next_population.extend([mutacao(c1), mutacao(c2)])

    population = next_population[:POP_SIZE]

fitnesses = [avaliar_hiperparametros(ind) for ind in population]
best_idx = int(np.argmax(fitnesses))
best_ind = population[best_idx]
best_params = decode(best_ind)
best_cv_f1 = float(fitnesses[best_idx])

print("\nMelhores hiperparâmetros encontrados:")
print(best_params)

model = RandomForestClassifier(
    **best_params,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
model.fit(X_train_fit, y_train)

pred = model.predict(X_test_fit)
proba = model.predict_proba(X_test_fit)[:, 1]

metricas = {
    "modelo": "RandomForest_GA_Hiperparametros",
    "n_features": X_train_fit.shape[1],
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "fitness_cv_f1": best_cv_f1,
}

logbook = pd.DataFrame(log_rows)
logbook.to_csv(HP_DIR / "logbook_hiperparametros.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(logbook["geracao"], logbook["fitness_media"], marker="o", label="Fitness média")
plt.plot(logbook["geracao"], logbook["fitness_maxima"], marker="o", label="Fitness máxima")
plt.xlabel("Geração")
plt.ylabel("F1-score em validação cruzada")
plt.title("Convergência do GA de Hiperparâmetros")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(HP_DIR / "convergencia_ga_hiperparametros.png", dpi=170)
plt.savefig(HP_DIR / "convergencia_ga_hiperparametros.pdf")
plt.close()

with open(HP_DIR / "best_hyperparams.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_params": best_params,
            "best_individual": best_ind,
            "metricas_teste": metricas,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

pd.DataFrame([metricas]).round(4).to_csv(HP_DIR / "metricas_random_forest_ga.csv", index=False)
pd.DataFrame([metricas]).round(4).to_markdown(HP_DIR / "metricas_random_forest_ga.md", index=False)
joblib.dump(model, HP_DIR / "modelo_random_forest_ga_hiperparametros.pkl")

md = f"""# GA de Hiperparâmetros

## Melhores hiperparâmetros

```json
{json.dumps(best_params, ensure_ascii=False, indent=2)}
```

## Métricas no conjunto de teste

{pd.DataFrame([metricas]).round(4).to_markdown(index=False)}

## Interpretação

O GA explorou combinações de hiperparâmetros do Random Forest. A fitness foi o F1-score médio em validação cruzada.
"""
(HP_DIR / "resumo_ga_hiperparametros.md").write_text(md, encoding="utf-8")

print("\nArtefactos guardados em:", HP_DIR)
print("Etapa 6 concluída com sucesso.")
