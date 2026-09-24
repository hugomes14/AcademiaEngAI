# Lab 06.3 — Séries Temporais
# Script 08 — Orquestrador
# Objetivo: executar todos os scripts pela ordem correta, com logs e seleção de etapas.

from pathlib import Path
import argparse
import subprocess
import sys
import time
import logging
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DADOS_DIR = BASE_DIR / "dados"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "execucao.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

parser = argparse.ArgumentParser(description="Orquestrador do Lab 06.3 — Séries Temporais")
parser.add_argument(
    "--etapas",
    default="1,2,3,4,5,6,7",
    help="Etapas a executar, separadas por vírgulas. Exemplo: 1,2,3",
)
args = parser.parse_args()

print("=" * 90)
print("ORQUESTRADOR — LAB 06.3: SÉRIES TEMPORAIS")
print("=" * 90)

# -------------------------------------------------------------------------
# Verificação dos datasets.
# Se não existirem, são criados datasets sintéticos simples para permitir
# a execução didática do laboratório.
# -------------------------------------------------------------------------
ficheiro_voltagem = DADOS_DIR / "voltagem_bateria.csv"
ficheiro_missoes = DADOS_DIR / "missoes_diarias.csv"
DADOS_DIR.mkdir(parents=True, exist_ok=True)

if not ficheiro_voltagem.exists():
    print("[AVISO] voltagem_bateria.csv não existe. A criar dataset sintético.")
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=14 * 24 * 60, freq="min")
    hora = idx.hour + idx.minute / 60
    voltagem = 3.7 + 0.12 * np.sin(2 * np.pi * hora / 24) + rng.normal(0, 0.04, len(idx))
    pd.DataFrame({"timestamp": idx, "voltagem": voltagem}).to_csv(ficheiro_voltagem, index=False)

if not ficheiro_missoes.exists():
    print("[AVISO] missoes_diarias.csv não existe. A criar dataset sintético.")
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=3 * 365, freq="D")
    tendencia = np.linspace(0, 8, len(idx))
    sazonalidade_semanal = np.where(idx.dayofweek < 5, 8, -3)
    sazonalidade_anual = 4 * np.sin(2 * np.pi * idx.dayofyear / 365)
    ruido = rng.normal(0, 3, len(idx))
    missoes = np.maximum(0, np.round(15 + tendencia + sazonalidade_semanal + sazonalidade_anual + ruido)).astype(int)
    pd.DataFrame({"data": idx, "num_missoes": missoes}).to_csv(ficheiro_missoes, index=False)

scripts = {
    "1": "01_analise_exploratoria.py",
    "2": "02_preprocessamento_features.py",
    "3": "03_treino_modelos.py",
    "4": "04_avaliacao_metricas.py",
    "5": "05_grafico_previsao_vs_real.py",
    "6": "06_analise_residuos.py",
    "7": "07_relatorio_automatico.py",
}

etapas = [e.strip() for e in args.etapas.split(",") if e.strip()]

print(f"[INFO] Etapas pedidas: {', '.join(etapas)}")
logging.info("Etapas pedidas: %s", etapas)

inicio_total = time.perf_counter()

for etapa in etapas:
    if etapa not in scripts:
        print(f"[AVISO] Etapa desconhecida: {etapa}. Ignorada.")
        logging.warning("Etapa desconhecida ignorada: %s", etapa)
        continue

    script = scripts[etapa]
    caminho_script = BASE_DIR / "scripts" / script

    print("\n" + "-" * 90)
    print(f"[EXECUÇÃO] Etapa {etapa}: {script}")
    print("-" * 90)

    logging.info("A iniciar etapa %s: %s", etapa, script)
    inicio = time.perf_counter()

    resultado = subprocess.run(
        [sys.executable, str(caminho_script)],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
    )

    duracao = time.perf_counter() - inicio

    print(resultado.stdout)
    if resultado.stderr:
        print("[STDERR]")
        print(resultado.stderr)

    if resultado.returncode != 0:
        print(f"[ERRO] A etapa {etapa} falhou com código {resultado.returncode}.")
        logging.error("Etapa %s falhou. STDERR: %s", etapa, resultado.stderr)

        decisao = "a"
        try:
            decisao = input("Escolha: [c]ontinuar, [a]bortar, [r]epetir? ").strip().lower()
        except EOFError:
            decisao = "a"

        if decisao == "r":
            print("[INFO] A repetir etapa uma vez.")
            resultado2 = subprocess.run(
                [sys.executable, str(caminho_script)],
                cwd=BASE_DIR,
                text=True,
                capture_output=True,
            )
            print(resultado2.stdout)
            if resultado2.stderr:
                print("[STDERR]")
                print(resultado2.stderr)
            if resultado2.returncode != 0:
                print("[ERRO] A repetição também falhou. Execução abortada.")
                sys.exit(resultado2.returncode)
        elif decisao == "c":
            print("[INFO] A continuar para a próxima etapa.")
            continue
        else:
            print("[INFO] Execução abortada.")
            sys.exit(resultado.returncode)
    else:
        print(f"[OK] Etapa {etapa} concluída em {duracao:.2f}s.")
        logging.info("Etapa %s concluída em %.2fs", etapa, duracao)

duracao_total = time.perf_counter() - inicio_total

print("\n" + "=" * 90)
print(f"[OK] Execução terminada em {duracao_total:.2f}s.")
print("Consulte artefactos/relatorios/RELATORIO_FINAL.md para o resultado final.")
print("=" * 90)

logging.info("Execução terminada em %.2fs", duracao_total)
