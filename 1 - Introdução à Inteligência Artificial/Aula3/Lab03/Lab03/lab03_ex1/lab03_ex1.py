# -*- coding: utf-8 -*-
"""Comparação de modelos de regressão para prever tempos de rally."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


FICHEIRO_DADOS = Path(__file__).with_name("dados_rally.csv")
VARIAVEIS = [
    "experiencia_anos",
    "potencia_cv",
    "comprimento_especial_km",
    "altimetria_total_m",
    "indice_aderencia",
]
ALVO = "tempo_especial_min"
GRAFICO_METRICAS = Path(__file__).with_name("comparacao_modelos.png")
GRAFICO_PREVISOES = Path(__file__).with_name("previsoes_melhor_modelo.png")


def carregar_dados(caminho):
    """Lê o CSV, exclui o identificador do piloto e prepara os valores numéricos."""
    with caminho.open(encoding="utf-8", newline="") as ficheiro:
        leitor = csv.DictReader(ficheiro)
        linhas = list(leitor)

    entradas = [
        [float(linha[variavel]) if linha[variavel] else math.nan for variavel in VARIAVEIS]
        for linha in linhas
    ]
    alvo = [float(linha[ALVO]) for linha in linhas]
    return entradas, alvo


def criar_pipeline(algoritmo):
    return Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="median")),
            ("normalizacao", StandardScaler()),
            ("modelo", algoritmo),
        ]
    )


def avaliar_modelos(entradas_treino, entradas_teste, alvo_treino, alvo_teste):
    algoritmos = {
        "Regressão Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "SVR": SVR(C=10.0, epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    resultados = []

    for nome, algoritmo in algoritmos.items():
        pipeline = criar_pipeline(algoritmo)
        pipeline.fit(entradas_treino, alvo_treino)
        previsoes = pipeline.predict(entradas_teste)
        resultados.append(
            {
                "nome": nome,
                "modelo": pipeline,
                "previsoes": previsoes,
                "rmse": math.sqrt(mean_squared_error(alvo_teste, previsoes)),
                "mae": mean_absolute_error(alvo_teste, previsoes),
                "r2": r2_score(alvo_teste, previsoes),
            }
        )

    return resultados


def adicionar_valores_barras(eixo, barras):
    for barra in barras:
        altura = barra.get_height()
        eixo.annotate(
            f"{altura:.3f}",
            (barra.get_x() + barra.get_width() / 2, altura),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def criar_grafico_metricas(resultados):
    nomes = [resultado["nome"] for resultado in resultados]
    metricas = [
        ("RMSE (min)", "rmse", "#d95f59"),
        ("MAE (min)", "mae", "#4c78a8"),
        ("R²", "r2", "#59a14f"),
    ]
    figura, eixos = plt.subplots(1, 3, figsize=(17, 5.5))

    for eixo, (titulo, chave, cor) in zip(eixos, metricas):
        valores = [resultado[chave] for resultado in resultados]
        barras = eixo.bar(nomes, valores, color=cor)
        eixo.set_title(titulo)
        eixo.tick_params(axis="x", rotation=35)
        eixo.grid(axis="y", alpha=0.25)
        adicionar_valores_barras(eixo, barras)

    figura.suptitle("Comparação dos modelos de previsão do tempo de rally")
    figura.tight_layout()
    figura.savefig(GRAFICO_METRICAS, dpi=160, bbox_inches="tight")
    plt.close(figura)


def criar_grafico_previsoes(alvo_teste, melhor_resultado):
    previsoes = melhor_resultado["previsoes"]
    limite_minimo = min(min(alvo_teste), min(previsoes))
    limite_maximo = max(max(alvo_teste), max(previsoes))

    figura, eixo = plt.subplots(figsize=(7, 6))
    eixo.scatter(alvo_teste, previsoes, alpha=0.75, color="#4c78a8")
    eixo.plot(
        [limite_minimo, limite_maximo],
        [limite_minimo, limite_maximo],
        linestyle="--",
        color="#d95f59",
        label="Previsão perfeita",
    )
    eixo.set_title(f"Valores reais vs. previstos - {melhor_resultado['nome']}")
    eixo.set_xlabel("Tempo real (min)")
    eixo.set_ylabel("Tempo previsto (min)")
    eixo.grid(alpha=0.25)
    eixo.legend()
    figura.tight_layout()
    figura.savefig(GRAFICO_PREVISOES, dpi=160, bbox_inches="tight")
    plt.close(figura)


def main():
    entradas, alvo = carregar_dados(FICHEIRO_DADOS)

    entradas_treino, entradas_teste, alvo_treino, alvo_teste = train_test_split(
        entradas, alvo, test_size=0.2, random_state=42
    )
    resultados = avaliar_modelos(
        entradas_treino, entradas_teste, alvo_treino, alvo_teste
    )
    resultados_ordenados = sorted(resultados, key=lambda resultado: resultado["r2"], reverse=True)
    melhor_resultado = resultados_ordenados[0]

    criar_grafico_metricas(resultados)
    criar_grafico_previsoes(alvo_teste, melhor_resultado)

    print("Comparação de modelos de regressão - dados de rally")
    print("Pré-processamento: imputação pela mediana e normalização das variáveis")
    print("Coluna excluída: nome_piloto (identificador)")
    print(f"Registos de treino: {len(alvo_treino)}")
    print(f"Registos de teste: {len(alvo_teste)}")
    print()
    print(f"{'Modelo':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 48)
    for resultado in resultados_ordenados:
        print(
            f"{resultado['nome']:<20} "
            f"{resultado['rmse']:>8.4f} "
            f"{resultado['mae']:>8.4f} "
            f"{resultado['r2']:>8.4f}"
        )

    exemplo = [[10, 280, 15, 300, 5]]
    tempo_previsto = melhor_resultado["modelo"].predict(exemplo)
    print(f"\nMelhor modelo por R²: {melhor_resultado['nome']}")
    print(f"Tempo previsto para o exemplo: {tempo_previsto[0]:.2f} minutos")
    print(f"Gráfico de métricas: {GRAFICO_METRICAS}")
    print(f"Gráfico de previsões: {GRAFICO_PREVISOES}")


if __name__ == "__main__":
    main()
