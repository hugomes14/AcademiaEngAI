# -*- coding: utf-8 -*-
"""
Orquestrador do Lab 06.5.

Executa os scripts pela ordem correta e regista a execução em logs/execucao.log.

Exemplos:
    python lab_orquestrador.py
    python lab_orquestrador.py --etapas 1 2 3
    python lab_orquestrador.py --parar-em-erro
"""

from pathlib import Path
import argparse
import subprocess
import sys
import time
import logging

RAIZ = Path(__file__).resolve().parent
LOGS = RAIZ / "logs"
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOGS / "execucao.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

ETAPAS = {
    1: ("Preparação dos dados", "01_preparacao_dados.py"),
    2: ("Treino PyTorch - 1 neurónio", "02_treino_pytorch_neuronio.py"),
    3: ("Treino sklearn - Regressão Linear", "03_treino_sklearn_comparacao.py"),
    4: ("Avaliação e métricas", "04_avaliacao_metricas.py"),
    5: ("Visualizações", "05_visualizacoes.py"),
    6: ("Relatório final", "06_relatorio_final.py"),
}

def cor(texto, codigo):
    return f"\033[{codigo}m{texto}\033[0m"

parser = argparse.ArgumentParser(description="Orquestrador do Lab 06.5")
parser.add_argument(
    "--etapas",
    nargs="*",
    type=int,
    choices=sorted(ETAPAS.keys()),
    help="Lista de etapas a executar. Se omitido, executa todas.",
)
parser.add_argument(
    "--parar-em-erro",
    action="store_true",
    help="Interrompe a execução no primeiro erro.",
)
args = parser.parse_args()

etapas_a_executar = args.etapas if args.etapas else sorted(ETAPAS.keys())

print(cor("=" * 80, "36"))
print(cor("LAB 06.5 - ORQUESTRADOR", "36"))
print(cor("=" * 80, "36"))
logging.info("Início da execução do orquestrador")

# Verificação básica do dataset.
dataset = RAIZ / "dados" / "voos_telemetria.csv"
if not dataset.exists():
    mensagem = f"Dataset não encontrado em {dataset}. Coloca o CSV na pasta dados/."
    logging.error(mensagem)
    print(cor(mensagem, "31"))
    sys.exit(1)

inicio_total = time.time()
falhas = []

for numero in etapas_a_executar:
    nome, script = ETAPAS[numero]
    caminho_script = RAIZ / script
    print("\n" + cor(f"▶ Etapa {numero}: {nome}", "33"))
    logging.info("A iniciar etapa %s - %s", numero, nome)

    inicio = time.time()
    resultado = subprocess.run(
        [sys.executable, str(caminho_script)],
        cwd=str(RAIZ),
        text=True,
    )
    duracao = time.time() - inicio

    if resultado.returncode == 0:
        print(cor(f"✓ Etapa {numero} concluída em {duracao:.2f}s", "32"))
        logging.info("Etapa %s concluída em %.2fs", numero, duracao)
    else:
        print(cor(f"✗ Etapa {numero} falhou com código {resultado.returncode}", "31"))
        logging.error("Etapa %s falhou com código %s", numero, resultado.returncode)
        falhas.append(numero)
        if args.parar_em_erro:
            break

duracao_total = time.time() - inicio_total
print("\n" + cor("=" * 80, "36"))
if falhas:
    print(cor(f"Execução terminada com falhas nas etapas: {falhas}", "31"))
else:
    print(cor("Execução concluída com sucesso.", "32"))
print(cor(f"Tempo total: {duracao_total:.2f}s", "36"))
print(cor("=" * 80, "36"))
logging.info("Fim da execução do orquestrador. Falhas=%s. Duração=%.2fs", falhas, duracao_total)

if falhas:
    sys.exit(1)
