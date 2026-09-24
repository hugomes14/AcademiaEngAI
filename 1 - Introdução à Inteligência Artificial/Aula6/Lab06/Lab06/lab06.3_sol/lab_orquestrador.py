# Atalho para executar o orquestrador principal.
# Pode correr:
#   python lab_orquestrador.py
# ou:
#   python scripts/08_lab_orquestrador.py

from pathlib import Path
import runpy

caminho = Path(__file__).resolve().parent / "scripts" / "08_lab_orquestrador.py"
runpy.run_path(str(caminho), run_name="__main__")
