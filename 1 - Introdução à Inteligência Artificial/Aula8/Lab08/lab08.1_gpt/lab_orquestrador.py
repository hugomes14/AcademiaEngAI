"""
Orquestrador — Lab 08.1 Aprendizagem por Reforço.

Executa os scripts pela ordem correta:
1. Exploração dos ambientes
2. Bandit
3. Treino Q-Learning
4. Avaliação
5. Gráficos de aprendizagem
6. Visualização da política
7. Relatório final

Exemplos:
    python lab_orquestrador.py
    python lab_orquestrador.py --etapas 1 2 3
    python lab_orquestrador.py --continuar-se-erro
"""

from pathlib import Path
import argparse
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "artefactos" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOGS_DIR / "execucao.log"

ETAPAS = {
    "1": "01_explora_ambientes.py",
    "2": "02_bandit_rotas.py",
    "3": "03_treino_qlearning.py",
    "4": "04_avaliacao_politica.py",
    "5": "05_graficos_aprendizagem.py",
    "6": "06_visualizacao_politica.py",
    "7": "07_relatorio_automatico.py",
}

parser = argparse.ArgumentParser(description="Orquestrador do Lab 08.1 — Aprendizagem por Reforço")
parser.add_argument(
    "--etapas",
    nargs="*",
    default=list(ETAPAS.keys()),
    help="Etapas a executar. Exemplo: --etapas 1 3 4",
)
parser.add_argument(
    "--continuar-se-erro",
    action="store_true",
    help="Continua para a etapa seguinte se uma etapa falhar.",
)
args = parser.parse_args()

with open(LOG_PATH, "a", encoding="utf-8") as log:
    log.write("\n" + "=" * 80 + "\n")
    log.write(f"Nova execução: {datetime.now().isoformat(timespec='seconds')}\n")

print("\n" + "=" * 80)
print("ORQUESTRADOR — LAB 08.1 APRENDIZAGEM POR REFORÇO")
print("=" * 80)
print(f"Diretório base: {BASE_DIR}")
print(f"Log: {LOG_PATH}")

tempo_total_inicio = time.perf_counter()

for etapa in args.etapas:
    if etapa not in ETAPAS:
        print(f"\n[AVISO] Etapa desconhecida ignorada: {etapa}")
        continue

    script = ETAPAS[etapa]
    script_path = BASE_DIR / script

    if not script_path.exists():
        mensagem = f"[ERRO] Script não encontrado: {script_path}"
        print(mensagem)
        with open(LOG_PATH, "a", encoding="utf-8") as log:
            log.write(mensagem + "\n")
        if not args.continuar_se_erro:
            sys.exit(1)
        continue

    print("\n" + "-" * 80)
    print(f"Etapa {etapa}: {script}")
    print("-" * 80)

    inicio = time.perf_counter()
    resultado = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        text=True,
        capture_output=True,
    )
    duracao = time.perf_counter() - inicio

    print(resultado.stdout)
    if resultado.stderr:
        print("[STDERR]")
        print(resultado.stderr)

    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\nEtapa {etapa}: {script}\n")
        log.write(f"Duração: {duracao:.2f}s\n")
        log.write(f"Código de saída: {resultado.returncode}\n")
        log.write(resultado.stdout)
        if resultado.stderr:
            log.write("\n[STDERR]\n")
            log.write(resultado.stderr)

    if resultado.returncode != 0:
        print(f"[ERRO] A etapa {etapa} falhou.")
        if not args.continuar_se_erro:
            print("Execução interrompida. Usa --continuar-se-erro para prosseguir mesmo com falhas.")
            sys.exit(resultado.returncode)

tempo_total = time.perf_counter() - tempo_total_inicio

print("\n" + "=" * 80)
print("EXECUÇÃO TERMINADA")
print("=" * 80)
print(f"Tempo total: {tempo_total:.2f}s")
print(f"Artefactos disponíveis em: {BASE_DIR / 'artefactos'}")
print(f"Relatório final: {BASE_DIR / 'artefactos' / 'relatorios' / 'RELATORIO_FINAL.md'}")
