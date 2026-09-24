# lab_orquestrador.py
# Executa as etapas do Lab 08.2 — Algoritmos Genéticos.

from pathlib import Path
import argparse
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "execucao.log"

ETAPAS = {
    "1": "01_gerar_dados_e_eda.py",
    "2": "02_preprocessamento.py",
    "3": "03_ga_selecao_features.py",
    "4": "04_grafico_convergencia_ga.py",
    "5": "05_avaliacao_melhor_subconjunto.py",
    "6": "06_ga_hiperparametros.py",
    "7": "07_nsga2_rotas.py",
    "8": "08_relatorio_final.py",
}

parser = argparse.ArgumentParser(description="Orquestrador do Lab 08.2 — Algoritmos Genéticos")
parser.add_argument(
    "--etapas",
    type=str,
    default="1,2,3,4,5,6,7,8",
    help="Lista de etapas a executar. Exemplo: --etapas 1,2,3",
)
parser.add_argument(
    "--continuar-em-erro",
    action="store_true",
    help="Continua para a etapa seguinte caso ocorra erro numa etapa.",
)
args = parser.parse_args()

selected = [e.strip() for e in args.etapas.split(",") if e.strip()]

print("=" * 80)
print("LAB 08.2 — ORQUESTRADOR")
print("=" * 80)
print("Etapas selecionadas:", ", ".join(selected))
print("Log:", LOG_FILE)

with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write("\n" + "=" * 80 + "\n")
    log.write("Nova execução do orquestrador\n")
    log.write("=" * 80 + "\n")

inicio_total = time.time()

for etapa in selected:
    if etapa not in ETAPAS:
        print(f"Etapa desconhecida: {etapa}")
        continue

    script = BASE_DIR / ETAPAS[etapa]
    print("\n" + "-" * 80)
    print(f"A executar etapa {etapa}: {script.name}")
    print("-" * 80)

    inicio = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    duracao = time.time() - inicio

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n[ETAPA {etapa}] {script.name}\n")
        log.write(f"Duração: {duracao:.2f}s\n")
        log.write("STDOUT:\n")
        log.write(result.stdout)
        log.write("\nSTDERR:\n")
        log.write(result.stderr)
        log.write("\n")

    print(result.stdout)
    if result.stderr:
        print("Mensagens de erro/aviso:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"ERRO: etapa {etapa} terminou com código {result.returncode}.")
        if not args.continuar_em_erro:
            print("Execução abortada. Usa --continuar-em-erro para prosseguir mesmo com erros.")
            sys.exit(result.returncode)
    else:
        print(f"Etapa {etapa} concluída em {duracao:.2f}s.")

duracao_total = time.time() - inicio_total
print("\n" + "=" * 80)
print(f"Execução concluída em {duracao_total:.2f}s.")
print("=" * 80)
