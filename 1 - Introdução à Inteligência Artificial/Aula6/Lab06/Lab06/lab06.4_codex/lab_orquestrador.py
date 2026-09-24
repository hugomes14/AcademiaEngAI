# lab_orquestrador.py
# Lab 06.4 — Modelos Ensemble
#
# Executa as etapas do laboratório pela ordem correta.
#
# Exemplos:
#   python lab_orquestrador.py
#   python lab_orquestrador.py --etapas eda preprocessamento treino
#   python lab_orquestrador.py --continuar-em-erro

from pathlib import Path
import argparse
import logging
import subprocess
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados"
LOG_DIR = BASE_DIR / "artefactos" / "logs"
DATASET_PATH = DATA_DIR / "voos_pre_voo.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "execucao.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

parser = argparse.ArgumentParser(description="Orquestrador do Lab 06.4 — Modelos Ensemble")
parser.add_argument(
    "--etapas",
    nargs="+",
    default=["eda", "preprocessamento", "treino", "avaliacao", "graficos", "explicabilidade", "relatorio"],
    choices=["eda", "preprocessamento", "treino", "avaliacao", "graficos", "explicabilidade", "relatorio"],
    help="Etapas a executar.",
)
parser.add_argument(
    "--continuar-em-erro",
    action="store_true",
    help="Continua para a etapa seguinte mesmo que uma etapa falhe.",
)
args = parser.parse_args()

print("=" * 80)
print("LAB 06.4 — ORQUESTRADOR")
print("=" * 80)

if not DATASET_PATH.exists():
    print("\nDataset não encontrado. A gerar dataset sintético compatível para demonstração.")
    rng = np.random.default_rng(42)
    n = 1000

    idade = rng.normal(10, 4, n).clip(1, 30)
    manutencao = rng.exponential(70, n).clip(1, 300)
    turbulencia = rng.choice(["Baixa", "Média", "Alta"], n, p=[0.55, 0.30, 0.15])
    missao = rng.choice(["Vigilância", "Carga", "Transporte"], n, p=[0.35, 0.30, 0.35])
    experiencia = rng.normal(12, 6, n).clip(1, 35)

    risco = (
        -4.0
        + 0.08 * idade
        + 0.010 * manutencao
        + 0.9 * (turbulencia == "Alta")
        + 0.4 * (turbulencia == "Média")
        + 0.35 * (missao == "Carga")
        - 0.05 * experiencia
    )
    prob = 1 / (1 + np.exp(-risco))
    incidente = rng.binomial(1, prob)

    df = pd.DataFrame({
        "idade_aeronave_anos": idade,
        "horas_voo_desde_ultima_manutencao": manutencao,
        "previsao_turbulencia": turbulencia,
        "tipo_missao": missao,
        "experiencia_piloto_anos": experiencia,
        "incidente_reportado": incidente,
    })

    # Ajuste simples para aproximar classe positiva de uma minoria.
    if df["incidente_reportado"].mean() > 0.20:
        positivos = df[df["incidente_reportado"] == 1].sample(frac=0.5, random_state=42).index
        df.loc[positivos, "incidente_reportado"] = 0

    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset sintético criado em: {DATASET_PATH}")

etapas = {
    "eda": "scripts/01_analise_exploratoria.py",
    "preprocessamento": "scripts/02_preprocessamento.py",
    "treino": "scripts/03_treino_ensembles.py",
    "avaliacao": "scripts/04_avaliacao_metricas.py",
    "graficos": "scripts/05_matriz_confusao_roc.py",
    "explicabilidade": "scripts/06_explicabilidade_pdp.py",
    "relatorio": "scripts/07_relatorio_automatico.py",
}

inicio_total = time.perf_counter()

for etapa in args.etapas:
    script = BASE_DIR / etapas[etapa]

    print("\n" + "-" * 80)
    print(f"A executar etapa: {etapa}")
    print(f"Script: {script}")
    logging.info("Início da etapa: %s", etapa)

    inicio = time.perf_counter()

    resultado = subprocess.run(
        [sys.executable, str(script)],
        cwd=BASE_DIR,
        text=True,
    )

    duracao = time.perf_counter() - inicio

    if resultado.returncode == 0:
        print(f"Etapa concluída: {etapa} ({duracao:.2f}s)")
        logging.info("Etapa concluída: %s | %.2fs", etapa, duracao)
    else:
        print(f"Erro na etapa: {etapa}")
        logging.error("Erro na etapa: %s | código %s", etapa, resultado.returncode)

        if not args.continuar_em_erro:
            print("Execução interrompida. Use --continuar-em-erro para tentar as restantes etapas.")
            sys.exit(resultado.returncode)

duracao_total = time.perf_counter() - inicio_total

print("\n" + "=" * 80)
print(f"Execução concluída em {duracao_total:.2f} segundos.")
print("Relatório final esperado em: artefactos/relatorios/RELATORIO_FINAL.md")
print("=" * 80)

logging.info("Execução completa em %.2fs", duracao_total)
