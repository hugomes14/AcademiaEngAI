"""Nome compatível com o enunciado; encaminha para a avaliação existente."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("04_avaliacao_metricas.py")), run_name="__main__")
