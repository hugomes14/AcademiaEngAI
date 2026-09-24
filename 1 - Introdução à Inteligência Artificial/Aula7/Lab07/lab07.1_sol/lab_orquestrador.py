"""Orquestrador do Lab 07.1 — clustering de perfis operacionais de voo."""

from pathlib import Path
import argparse
import csv
from datetime import datetime
import logging
import shutil
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "voos_telemetria_completa.csv"
CANONICAL_DATASET = BASE_DIR.parent / "lab07.1" / "voos_telemetria_completa.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "matplotlib_cache").mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "execucao.log"

ETAPAS = {
    "1": ("Análise Exploratória", BASE_DIR / "scripts" / "01_analise_exploratoria.py"),
    "2": ("Pré-processamento", BASE_DIR / "scripts" / "02_preprocessamento.py"),
    "3": ("Método Elbow", BASE_DIR / "scripts" / "03_metodo_elbow.py"),
    "4": ("Treino e Avaliação", BASE_DIR / "scripts" / "04_treino_avaliacao.py"),
    "5": ("Visualização PCA", BASE_DIR / "scripts" / "05_visualizacao_pca.py"),
    "6": ("DBSCAN e Perfil dos Clusters", BASE_DIR / "scripts" / "06_dbscan_perfil_clusters.py"),
    "7": ("Relatório Automático", BASE_DIR / "scripts" / "07_relatorio_automatico.py"),
}

parser = argparse.ArgumentParser(description="Orquestrador do Lab 07.1 — Clustering")
parser.add_argument("--listar", action="store_true", help="Lista as etapas sem as executar.")
parser.add_argument("--etapa", choices=ETAPAS, help="Executa apenas uma etapa.")
parser.add_argument("--etapas", nargs="+", choices=ETAPAS, help="Compatibilidade: executa várias etapas.")
parser.add_argument("--de", dest="inicio", type=int, choices=range(1, 8), help="Primeira etapa do intervalo.")
parser.add_argument("--ate", dest="fim", type=int, choices=range(1, 8), help="Última etapa do intervalo.")
parser.add_argument("--continuar-em-erro", "--continuar-se-erro", action="store_true", dest="continuar", help="Continua após uma falha.")
parser.add_argument("--tentativas", type=int, default=1, help="Número máximo de tentativas por etapa (predefinição: 1).")
parser.add_argument("--gerar-dados", action="store_true", help="Gera dados sintéticos apenas se o dataset original faltar.")
parser.add_argument("--sem-cor", action="store_true", help="Desativa cores no terminal.")
args = parser.parse_args()

if args.tentativas < 1:
    parser.error("--tentativas deve ser pelo menos 1.")
if args.etapa and args.etapas:
    parser.error("Escolhe --etapa ou --etapas, não ambos.")
if (args.inicio is None) != (args.fim is None):
    parser.error("--de e --ate devem ser usados em conjunto.")
if args.inicio is not None and args.inicio > args.fim:
    parser.error("--de não pode ser maior do que --ate.")

if args.etapa:
    ids = [args.etapa]
elif args.etapas:
    ids = args.etapas
elif args.inicio is not None:
    ids = [str(numero) for numero in range(args.inicio, args.fim + 1)]
else:
    ids = list(ETAPAS)

if args.listar:
    print("Etapas do Lab 07.1:")
    for etapa_id, (nome, script) in ETAPAS.items():
        print(f"  {etapa_id}. {nome} — {script.relative_to(BASE_DIR)}")
    sys.exit(0)

def escrever_log(mensagem):
    linha = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mensagem}"
    print(linha)
    with LOG_PATH.open("a", encoding="utf-8") as ficheiro:
        ficheiro.write(linha + "\n")

def colorir(texto, codigo):
    return texto if args.sem_cor else f"\033[{codigo}m{texto}\033[0m"

def gerar_dataset_sintetico():
    """Cria um recurso local explícito; nunca substitui silenciosamente o dataset real."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    blocos = []
    for centro, escala, n in [((25, 20, 150, 55, 5, 300), (4, 4, 25, 8, 1, 50), 100),
                              ((120, 200, 5000, 180, 40, 1000), (12, 20, 500, 18, 5, 150), 100),
                              ((45, 50, 1100, 120, 15, 8000), (6, 6, 120, 12, 2, 500), 100)]:
        blocos.append(rng.normal(centro, escala, size=(n, 6)))
    dados = np.vstack(blocos)
    dados = np.maximum(dados, 0.01)
    pd.DataFrame(dados, columns=["duracao_voo_min", "distancia_percorrida_km", "altitude_maxima_m",
                                 "velocidade_media_kmh", "consumo_combustivel_litros",
                                 "variacao_vertical_total_m"]).to_csv(DATASET_PATH, index=False)
    DATASET_PATH.with_name("DADOS_SINTETICOS.md").write_text(
        "# Dados sintéticos\n\nGerados com seed 42 apenas como recurso de execução; não representam o dataset original.\n",
        encoding="utf-8")

if not DATASET_PATH.exists():
    if CANONICAL_DATASET.exists():
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CANONICAL_DATASET, DATASET_PATH)
        escrever_log(f"Dataset copiado da origem canónica: {CANONICAL_DATASET}")
    elif args.gerar_dados:
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        gerar_dataset_sintetico()
        escrever_log(f"Dataset sintético criado: {DATASET_PATH}")
    else:
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_PATH}. Usa --gerar-dados apenas como recurso explícito.")

faltam = [str(path) for _, path in (ETAPAS[etapa_id] for etapa_id in ids) if not path.exists()]
if faltam:
    raise FileNotFoundError("Scripts em falta:\n" + "\n".join(faltam))

logging.basicConfig(filename=LOG_PATH, level=logging.INFO, encoding="utf-8")
escrever_log(f"Início da execução. Dataset: {DATASET_PATH}")
escrever_log(f"Etapas selecionadas: {', '.join(ids)}")

resumos = []
inicio_total = time.time()
falhas = []
for etapa_id in ids:
    nome, script_path = ETAPAS[etapa_id]
    sucesso = False
    ultima_saida = ""
    for tentativa in range(1, args.tentativas + 1):
        inicio_etapa = time.time()
        print("\n" + "-" * 90)
        print(colorir(f"ETAPA {etapa_id}: {nome} — tentativa {tentativa}/{args.tentativas}", "96"))
        print("-" * 90)
        ambiente = dict(__import__("os").environ)
        ambiente.update({"MPLCONFIGDIR": str(OUTPUT_DIR / "matplotlib_cache"), "OMP_NUM_THREADS": "1",
                         "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        resultado = subprocess.run([sys.executable, str(script_path)], cwd=BASE_DIR, text=True,
                                   capture_output=True, env=ambiente)
        duracao = time.time() - inicio_etapa
        ultima_saida = (resultado.stdout or "") + (resultado.stderr or "")
        if resultado.stdout:
            print(resultado.stdout)
        if resultado.stderr:
            print(colorir("[STDERR]", "93"))
            print(resultado.stderr)
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\nETAPA {etapa_id} — tentativa {tentativa} — duração {duracao:.2f}s\n{ultima_saida}\n")
        if resultado.returncode == 0:
            sucesso = True
            escrever_log(f"Etapa {etapa_id} concluída em {duracao:.2f}s.")
            break
        escrever_log(f"Etapa {etapa_id} falhou com código {resultado.returncode}.")
    resumos.append({"etapa": etapa_id, "nome": nome, "estado": "sucesso" if sucesso else "falha",
                    "tentativas": tentativa, "duracao_segundos": round(duracao, 3)})
    if not sucesso:
        falhas.append(etapa_id)
        if not args.continuar:
            break

tempo_total = time.time() - inicio_total
with (LOG_DIR / "resumo_execucao.csv").open("w", newline="", encoding="utf-8") as ficheiro:
    escritor = csv.DictWriter(ficheiro, fieldnames=["etapa", "nome", "estado", "tentativas", "duracao_segundos"])
    escritor.writeheader()
    escritor.writerows(resumos)
linhas_md = ["# Resumo da execução — Lab 07.1", "", "| Etapa | Nome | Estado | Tentativas | Duração (s) |", "|---:|---|---|---:|---:|"]
linhas_md.extend(f"| {r['etapa']} | {r['nome']} | {r['estado']} | {r['tentativas']} | {r['duracao_segundos']:.3f} |" for r in resumos)
linhas_md.extend(["", f"Tempo total: {tempo_total:.3f} s", f"Etapas com falha: {', '.join(falhas) if falhas else 'nenhuma'}"])
(LOG_DIR / "resumo_execucao.md").write_text("\n".join(linhas_md) + "\n", encoding="utf-8")
escrever_log(f"Fim da execução. Duração total: {tempo_total:.2f}s.")
print(colorir("EXECUÇÃO CONCLUÍDA COM FALHAS" if falhas else "EXECUÇÃO COMPLETA COM SUCESSO", "91" if falhas else "92"))
print(f"Resumo: {LOG_DIR / 'resumo_execucao.md'}")
print(f"Log: {LOG_PATH}")
sys.exit(1 if falhas else 0)
