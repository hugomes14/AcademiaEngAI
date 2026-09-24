# lab_orquestrador.py
# Lab 07.2 — Redução de Dimensionalidade
# Executa as etapas do projeto na ordem correta.
#
# Exemplos:
#   python lab_orquestrador.py --listar
#   python lab_orquestrador.py
#   python lab_orquestrador.py --etapa 03
#   python lab_orquestrador.py --desde 03 --ate 05
#   python lab_orquestrador.py --tentativas 2 --continuar

from pathlib import Path
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
MODELOS_DIR = BASE_DIR / "modelos"
TABELAS_DIR = BASE_DIR / "tabelas"
IMAGENS_DIR = BASE_DIR / "imagens"
RELATORIOS_DIR = BASE_DIR / "relatorios"
LOGS_DIR = BASE_DIR / "logs"
MPLCONFIG_DIR = BASE_DIR / ".matplotlib_cache"
NUMBA_CACHE_DIR = BASE_DIR / ".numba_cache"

for directory in [
    DADOS_DIR,
    ARTEFACTOS_DIR,
    MODELOS_DIR,
    TABELAS_DIR,
    IMAGENS_DIR,
    RELATORIOS_DIR,
    LOGS_DIR,
    MPLCONFIG_DIR,
    NUMBA_CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DADOS_DIR / "sensores_voo_alta_dim.csv"
DATASET_ORIGEM = BASE_DIR.parent / "lab07.2" / "sensores_voo_alta_dim.csv"
LOG_PATH = LOGS_DIR / "execucao.log"
RESUMO_CSV_PATH = LOGS_DIR / "resumo_execucao.csv"
RELATORIO_PATH = RELATORIOS_DIR / "RELATORIO_FINAL.md"

SCRIPTS = [
    ("01", "01_analise_exploratoria.py", "Análise exploratória"),
    ("02", "02_preprocessamento.py", "Pré-processamento"),
    ("03", "03_pca_selecao_componentes.py", "PCA — seleção de componentes"),
    ("04", "04_pca_transformacao_loadings.py", "PCA — transformação e loadings"),
    ("05", "05_visualizacao_pca_reconstrucao.py", "Visualização PCA e reconstrução"),
    ("06", "06_visualizacao_nao_linear_lda.py", "t-SNE, Kernel PCA, UMAP e LDA"),
    ("07", "07_relatorio_automatico.py", "Relatório automático"),
]

codigos_validos = [codigo for codigo, _, _ in SCRIPTS]

parser = argparse.ArgumentParser(
    description="Orquestrador do Lab 07.2 — Redução de Dimensionalidade"
)
parser.add_argument("--listar", action="store_true", help="Lista as etapas disponíveis e termina.")
parser.add_argument("--etapa", choices=codigos_validos, help="Executa apenas uma etapa específica.")
parser.add_argument("--desde", choices=codigos_validos, help="Executa desde uma etapa específica.")
parser.add_argument("--de", choices=codigos_validos, help="Alias de --desde.")
parser.add_argument("--ate", choices=codigos_validos, help="Executa até uma etapa específica.")
parser.add_argument("--continuar", action="store_true", help="Continua a execução mesmo que uma etapa falhe.")
parser.add_argument("--continuar-se-erro", action="store_true", help="Alias de --continuar.")
parser.add_argument("--tentativas", type=int, default=1, help="Número de tentativas por etapa. Valor mínimo: 1.")
args = parser.parse_args()

continuar_apos_erro = args.continuar or args.continuar_se_erro
tentativas = max(1, args.tentativas)
desde = args.desde or args.de

print("=" * 80)
print("LAB 07.2 — ORQUESTRADOR")
print("=" * 80)

if args.listar:
    print("Etapas disponíveis:")
    for codigo, script, descricao in SCRIPTS:
        print(f"- {codigo}: {descricao} ({script})")
    sys.exit(0)

if args.etapa and (desde or args.ate):
    print("Erro: usa --etapa sozinho ou combina --desde/--de com --ate.")
    sys.exit(2)

if not DATASET_PATH.exists():
    if DATASET_ORIGEM.exists():
        shutil.copy2(DATASET_ORIGEM, DATASET_PATH)
        print(f"Dataset copiado de {DATASET_ORIGEM} para {DATASET_PATH}.")
    else:
        print(f"Dataset não encontrado em {DATASET_PATH}.")
        print(f"Também não foi encontrado em {DATASET_ORIGEM}.")
        print("Coloca `sensores_voo_alta_dim.csv` na pasta `dados` e volta a executar.")
        sys.exit(1)

if args.etapa:
    scripts_a_executar = [item for item in SCRIPTS if item[0] == args.etapa]
else:
    inicio = 0
    fim = len(SCRIPTS) - 1

    if desde:
        inicio = codigos_validos.index(desde)
    if args.ate:
        fim = codigos_validos.index(args.ate)

    if inicio > fim:
        print("Erro: a etapa inicial não pode ficar depois da etapa final.")
        sys.exit(2)

    scripts_a_executar = SCRIPTS[inicio:fim + 1]

inicio_global = time.time()
falhas = []
resumo_linhas = []

with LOG_PATH.open("a", encoding="utf-8") as log:
    log.write("\n" + "=" * 80 + "\n")
    log.write(f"Nova execução: {datetime.now().isoformat(timespec='seconds')}\n")
    log.write(f"Python: {sys.executable}\n")
    log.write(f"Dataset: {DATASET_PATH}\n")
    log.write(f"Tentativas por etapa: {tentativas}\n")
    log.write(f"Continuar após erro: {continuar_apos_erro}\n")
    log.write(f"Scripts: {[script for _, script, _ in scripts_a_executar]}\n")

for codigo, script, descricao in scripts_a_executar:
    script_path = BASE_DIR / script
    print("\n" + "-" * 80)
    print(f"Etapa {codigo}: {descricao}")
    print(f"Script: {script}")
    print("-" * 80)

    if not script_path.exists():
        mensagem = f"Script em falta: {script_path}"
        print(mensagem)
        falhas.append((codigo, mensagem))
        resumo_linhas.append({
            "etapa": codigo,
            "descricao": descricao,
            "script": script,
            "estado": "falhou",
            "tentativa": 0,
            "duracao_segundos": "0.00",
            "codigo_retorno": 1,
        })
        if not continuar_apos_erro:
            break
        continue

    etapa_sucesso = False
    codigo_retorno_final = 1
    duracao_final = 0.0

    for tentativa in range(1, tentativas + 1):
        print(f"Tentativa {tentativa}/{tentativas}")
        inicio = time.time()
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
        env["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_DIR)
        resultado = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            text=True,
            capture_output=True,
            env=env,
        )
        duracao = time.time() - inicio
        codigo_retorno_final = resultado.returncode
        duracao_final = duracao

        print(resultado.stdout)
        if resultado.stderr:
            print("STDERR:")
            print(resultado.stderr)

        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Etapa {codigo} — {descricao}\n")
            log.write(f"Tentativa: {tentativa}/{tentativas}\n")
            log.write(f"Comando: {sys.executable} {script_path}\n")
            log.write(f"Duração: {duracao:.2f}s\n")
            log.write(f"Código de retorno: {resultado.returncode}\n")
            log.write("STDOUT:\n")
            log.write(resultado.stdout)
            if resultado.stderr:
                log.write("\nSTDERR:\n")
                log.write(resultado.stderr)

        if resultado.returncode == 0:
            etapa_sucesso = True
            print(f"Etapa {codigo} concluída em {duracao:.2f}s.")
            resumo_linhas.append({
                "etapa": codigo,
                "descricao": descricao,
                "script": script,
                "estado": "sucesso",
                "tentativa": tentativa,
                "duracao_segundos": f"{duracao:.2f}",
                "codigo_retorno": resultado.returncode,
            })
            break

        print(f"Etapa {codigo} falhou com código {resultado.returncode}.")

    if not etapa_sucesso:
        mensagem = f"Etapa {codigo} falhou após {tentativas} tentativa(s)."
        falhas.append((codigo, mensagem))
        resumo_linhas.append({
            "etapa": codigo,
            "descricao": descricao,
            "script": script,
            "estado": "falhou",
            "tentativa": tentativas,
            "duracao_segundos": f"{duracao_final:.2f}",
            "codigo_retorno": codigo_retorno_final,
        })
        if not continuar_apos_erro:
            print("Execução interrompida. Use --continuar para seguir após falhas.")
            break

with RESUMO_CSV_PATH.open("w", encoding="utf-8", newline="") as resumo_csv:
    campos = ["etapa", "descricao", "script", "estado", "tentativa", "duracao_segundos", "codigo_retorno"]
    writer = csv.DictWriter(resumo_csv, fieldnames=campos)
    writer.writeheader()
    writer.writerows(resumo_linhas)

duracao_global = time.time() - inicio_global

print("\n" + "=" * 80)
print("RESUMO DA EXECUÇÃO")
print("=" * 80)
print(f"Duração total: {duracao_global:.2f}s")
print(f"Log: {LOG_PATH}")
print(f"Resumo CSV: {RESUMO_CSV_PATH}")

if RELATORIO_PATH.exists():
    print(f"Relatório final: {RELATORIO_PATH}")
else:
    print("Relatório final ainda não existe.")

if falhas:
    print("\nFalhas registadas:")
    for codigo, mensagem in falhas:
        print(f"- Etapa {codigo}: {mensagem}")
    sys.exit(1)

print("\nTodas as etapas selecionadas foram concluídas com sucesso.")
