# lab_orquestrador.py
# Lab 06.1 - Regressão — Estimar a Duração de Voo
# Objetivo: executar os scripts do laboratório pela ordem correta, com logs e controlo de etapas.

from pathlib import Path
import argparse
import subprocess
import sys
import time
import pandas as pd
import numpy as np

# -----------------------------
# Configuração base
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "voos_telemetria.csv"
SCRIPTS_DIR = BASE_DIR / "scripts"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_PATH = OUTPUT_DIR / "execucao.log"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

# -----------------------------
# Argumentos de execução
# -----------------------------
parser = argparse.ArgumentParser(description="Orquestrador do Lab 06.1 - Regressão de Duração de Voo")
parser.add_argument(
    "--etapas",
    nargs="+",
    default=["todas"],
    help=(
        "Etapas a executar: todas, 01, 02, 03, 04, 05, 06, 07. "
        "Exemplo: python lab_orquestrador.py --etapas 01 02 03"
    )
)
parser.add_argument(
    "--continuar-em-erro",
    action="store_true",
    help="Continua para a etapa seguinte mesmo que uma etapa falhe."
)
parser.add_argument(
    "--tentativas",
    type=int,
    default=1,
    help="Número de tentativas por etapa antes de abortar ou avançar."
)
args = parser.parse_args()

# -----------------------------
# Verificação ou geração do dataset
# -----------------------------
print(f"{BLUE}A verificar dataset em {DATASET_PATH}{RESET}")

if not DATASET_PATH.exists():
    print(f"{YELLOW}Dataset não encontrado. Será gerado um dataset sintético compatível com o laboratório.{RESET}")
    rng = np.random.default_rng(42)
    n_rows = 500
    distancia = rng.uniform(300, 5000, n_rows).round(2)
    carga = rng.uniform(500, 10000, n_rows).round(2)
    altitude = rng.uniform(7000, 12000, n_rows).round(2)
    meteo = rng.choice(["Bom", "Moderado", "Adverso"], size=n_rows, p=[0.55, 0.30, 0.15])
    meteo_impact = pd.Series(meteo).map({"Bom": 0, "Moderado": 12, "Adverso": 28}).to_numpy()
    ruido = rng.normal(0, 12, n_rows)
    duracao = 25 + distancia * 0.12 + carga * 0.002 + (12000 - altitude) * 0.003 + meteo_impact + ruido
    duracao = np.maximum(duracao, 20).round(2)
    synthetic_df = pd.DataFrame({
        "distancia_planeada": distancia,
        "carga_util_kg": carga,
        "altitude_media_m": altitude,
        "condicao_meteo": meteo,
        "duracao_voo_min": duracao
    })
    synthetic_df.to_csv(DATASET_PATH, index=False)
    print(f"{GREEN}Dataset sintético criado em {DATASET_PATH}{RESET}")
else:
    print(f"{GREEN}Dataset encontrado.{RESET}")

# -----------------------------
# Lista de scripts disponíveis
# -----------------------------
all_steps = [
    ("01", SCRIPTS_DIR / "01_analise_exploratoria.py"),
    ("02", SCRIPTS_DIR / "02_preprocessamento.py"),
    ("03", SCRIPTS_DIR / "03_treino_modelos.py"),
    ("04", SCRIPTS_DIR / "04_avaliacao_metricas.py"),
    ("05", SCRIPTS_DIR / "05_grafico_previsto_vs_real.py"),
    ("06", SCRIPTS_DIR / "06_analise_residuos.py"),
    ("07", SCRIPTS_DIR / "07_relatorio_automatico.py")
]

requested_steps = [step.lower() for step in args.etapas]
if "todas" in requested_steps or "all" in requested_steps:
    steps_to_run = all_steps
else:
    valid_codes = {code for code, _ in all_steps}
    invalid_codes = [step for step in requested_steps if step not in valid_codes]
    if invalid_codes:
        raise ValueError(f"Etapas inválidas: {invalid_codes}. Usa: {sorted(valid_codes)} ou 'todas'.")
    steps_to_run = [(code, path) for code, path in all_steps if code in requested_steps]

for code, script_path in steps_to_run:
    if not script_path.exists():
        raise FileNotFoundError(f"Script da etapa {code} não encontrado: {script_path}")

# -----------------------------
# Execução sequencial com log
# -----------------------------
start_all = time.perf_counter()
log_lines = []
log_lines.append("=" * 80 + "\n")
log_lines.append("Execução do Lab 06.1 - Regressão — Estimar a Duração de Voo\n")
log_lines.append(f"Etapas solicitadas: {[code for code, _ in steps_to_run]}\n")
log_lines.append(f"Continuar em erro: {args.continuar_em_erro}\n")
log_lines.append(f"Tentativas por etapa: {args.tentativas}\n")
log_lines.append("=" * 80 + "\n")

failures = []

for code, script_path in steps_to_run:
    print("\n" + "=" * 80)
    print(f"{BLUE}A executar etapa {code}: {script_path.name}{RESET}")
    print("=" * 80)

    attempt = 1
    step_success = False
    step_start = time.perf_counter()

    while attempt <= args.tentativas and not step_success:
        print(f"{YELLOW}Tentativa {attempt}/{args.tentativas}{RESET}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True
        )

        log_lines.append(f"\n--- ETAPA {code} | {script_path.name} | TENTATIVA {attempt} ---\n")
        log_lines.append("STDOUT:\n")
        log_lines.append(result.stdout + "\n")
        log_lines.append("STDERR:\n")
        log_lines.append(result.stderr + "\n")
        log_lines.append(f"RETURN_CODE: {result.returncode}\n")

        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(f"{RED}{result.stderr}{RESET}")

        if result.returncode == 0:
            step_success = True
            elapsed = time.perf_counter() - step_start
            print(f"{GREEN}Etapa {code} concluída em {elapsed:.2f} segundos.{RESET}")
        else:
            print(f"{RED}Etapa {code} falhou com código {result.returncode}.{RESET}")
            attempt += 1
            if attempt <= args.tentativas:
                print(f"{YELLOW}Nova tentativa preparada.{RESET}")

    if not step_success:
        failures.append(code)
        if not args.continuar_em_erro:
            print(f"{RED}Execução abortada na etapa {code}.{RESET}")
            break
        print(f"{YELLOW}A continuar apesar da falha na etapa {code}.{RESET}")

elapsed_all = time.perf_counter() - start_all
log_lines.append("\n" + "=" * 80 + "\n")
log_lines.append(f"Tempo total: {elapsed_all:.2f} segundos\n")
log_lines.append(f"Falhas: {failures}\n")
LOG_PATH.write_text("".join(log_lines), encoding="utf-8")

print("\n" + "=" * 80)
if failures:
    print(f"{RED}Execução terminada com falhas nas etapas: {failures}{RESET}")
else:
    print(f"{GREEN}Execução concluída com sucesso.{RESET}")
print(f"Log guardado em: {LOG_PATH}")
print(f"Tempo total: {elapsed_all:.2f} segundos")
print("=" * 80)
