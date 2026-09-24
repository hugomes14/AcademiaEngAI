# -*- coding: utf-8 -*-
"""
Lab 06.2 - Classificação — Prever o Risco de Incidentes em Voos
Ficheiro Orquestrador

Executa as etapas do laboratório pela ordem correta:
1. EDA
2. Pré-processamento
3. Treino
4. Avaliação
5. Matriz de Confusão
6. Curvas ROC / Precision-Recall
7. Relatório Final

Exemplos:
    python lab_orquestrador.py
    python lab_orquestrador.py --etapas 1 2 3
    python lab_orquestrador.py --continuar-se-erro
    python lab_orquestrador.py --gerar-dataset-se-faltar
"""

from pathlib import Path
import argparse
import subprocess
import sys
import time
import logging

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "voos_pre_voo.csv"
LOG_PATH = BASE_DIR / "execucao.log"

ETAPAS = {
    "1": BASE_DIR / "scripts" / "01_analise_exploratoria.py",
    "2": BASE_DIR / "scripts" / "02_pre_processamento.py",
    "3": BASE_DIR / "scripts" / "03_treino_modelos.py",
    "4": BASE_DIR / "scripts" / "04_avaliacao_metricas.py",
    "5": BASE_DIR / "scripts" / "05_matriz_confusao.py",
    "6": BASE_DIR / "scripts" / "06_curvas_roc_pr.py",
    "7": BASE_DIR / "scripts" / "07_relatorio_automatico.py",
}

CORES = {
    "verde": "\033[92m",
    "amarelo": "\033[93m",
    "vermelho": "\033[91m",
    "azul": "\033[94m",
    "fim": "\033[0m",
}

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

def cor(texto, nome):
    return f"{CORES.get(nome, '')}{texto}{CORES['fim']}"


def gerar_dataset_sintetico(path):
    """Gera um dataset sintético compatível, apenas se o ficheiro original faltar."""
    rng = np.random.default_rng(42)
    n = 1000

    idade = np.clip(rng.gamma(shape=2.2, scale=3.8, size=n), 0, 30)
    manutencao = np.clip(rng.gamma(shape=2.0, scale=45.0, size=n), 0, 400)
    experiencia = np.clip(rng.normal(loc=9, scale=5, size=n), 0, 35)
    turbulencia = rng.choice(["Baixa", "Média", "Alta"], size=n, p=[0.62, 0.32, 0.06])
    missao = rng.choice(["Transporte", "Carga", "Vigilância"], size=n, p=[0.55, 0.33, 0.12])

    turb_score = pd.Series(turbulencia).map({"Baixa": 0, "Média": 1, "Alta": 2}).to_numpy()
    missao_score = pd.Series(missao).map({"Transporte": 0.0, "Carga": 0.35, "Vigilância": 0.2}).to_numpy()

    logits = (
        -3.5
        + 0.07 * idade
        + 0.006 * manutencao
        - 0.08 * experiencia
        + 0.9 * turb_score
        + missao_score
    )
    probs = 1 / (1 + np.exp(-logits))
    incidente = rng.binomial(1, probs)

    # Ajuste simples para manter a classe positiva relativamente rara.
    if incidente.mean() > 0.18:
        limiar = np.quantile(probs, 0.90)
        incidente = (probs >= limiar).astype(int)

    df = pd.DataFrame({
        "idade_aeronave_anos": idade,
        "horas_voo_desde_ultima_manutencao": manutencao,
        "previsao_turbulencia": turbulencia,
        "tipo_missao": missao,
        "experiencia_piloto_anos": experiencia,
        "incidente_reportado": incidente,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(cor(f"Dataset sintético criado em {path}", "amarelo"))


def executar_etapa(numero, script_path):
    print(cor("\n" + "=" * 80, "azul"))
    print(cor(f"A executar etapa {numero}: {script_path.name}", "azul"))
    print(cor("=" * 80, "azul"))

    inicio = time.perf_counter()
    resultado = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        text=True,
    )
    duracao = time.perf_counter() - inicio

    if resultado.returncode == 0:
        mensagem = f"Etapa {numero} concluída em {duracao:.2f} segundos."
        print(cor(mensagem, "verde"))
        logging.info(mensagem)
        return True

    mensagem = f"Etapa {numero} falhou com código {resultado.returncode} após {duracao:.2f} segundos."
    print(cor(mensagem, "vermelho"))
    logging.error(mensagem)
    return False


def pedir_decisao():
    print("\nEscolhe uma opção:")
    print("1 — continuar para a próxima etapa")
    print("2 — abortar execução")
    print("3 — tentar a mesma etapa novamente")
    escolha = input("Opção [1/2/3]: ").strip()
    return escolha


def main():
    parser = argparse.ArgumentParser(description="Orquestrador do Lab 06.2 — Classificação de risco de incidentes em voos")
    parser.add_argument(
        "--etapas",
        nargs="*",
        default=list(ETAPAS.keys()),
        help="Lista de etapas a executar. Exemplo: --etapas 1 2 3",
    )
    parser.add_argument(
        "--continuar-se-erro",
        action="store_true",
        help="Continua a execução mesmo quando uma etapa falha.",
    )
    parser.add_argument(
        "--gerar-dataset-se-faltar",
        action="store_true",
        help="Gera um dataset sintético compatível se data/voos_pre_voo.csv não existir.",
    )
    args = parser.parse_args()

    print(cor("Lab 06.2 — Classificação: Prever o Risco de Incidentes em Voos", "azul"))
    print(f"Diretório base: {BASE_DIR}")
    print(f"Log: {LOG_PATH}")

    if not DATASET_PATH.exists():
        if args.gerar_dataset_se_faltar:
            gerar_dataset_sintetico(DATASET_PATH)
        else:
            print(cor(f"Dataset não encontrado: {DATASET_PATH}", "vermelho"))
            print("Coloca o ficheiro voos_pre_voo.csv na pasta data/ ou usa --gerar-dataset-se-faltar.")
            sys.exit(1)

    etapas_invalidas = [etapa for etapa in args.etapas if etapa not in ETAPAS]
    if etapas_invalidas:
        print(cor(f"Etapas inválidas: {etapas_invalidas}", "vermelho"))
        print(f"Etapas válidas: {list(ETAPAS.keys())}")
        sys.exit(1)

    inicio_total = time.perf_counter()
    logging.info("Início da execução do laboratório.")

    for etapa in args.etapas:
        script = ETAPAS[etapa]
        if not script.exists():
            print(cor(f"Script não encontrado: {script}", "vermelho"))
            sys.exit(1)

        while True:
            sucesso = executar_etapa(etapa, script)
            if sucesso:
                break

            if args.continuar_se_erro:
                print(cor("Erro ignorado devido a --continuar-se-erro.", "amarelo"))
                break

            decisao = pedir_decisao()
            if decisao == "1":
                break
            if decisao == "2":
                print(cor("Execução abortada pelo utilizador.", "vermelho"))
                sys.exit(1)
            if decisao == "3":
                print(cor("A tentar novamente a mesma etapa...", "amarelo"))
                continue

            print(cor("Opção inválida. A execução será abortada por segurança.", "vermelho"))
            sys.exit(1)

    duracao_total = time.perf_counter() - inicio_total
    logging.info(f"Execução concluída em {duracao_total:.2f} segundos.")
    print(cor("\nExecução concluída com sucesso.", "verde"))
    print(f"Tempo total: {duracao_total:.2f} segundos")
    print(f"Relatório esperado: {BASE_DIR / 'artefactos' / 'relatorios' / 'RELATORIO_FINAL.md'}")


if __name__ == "__main__":
    main()
