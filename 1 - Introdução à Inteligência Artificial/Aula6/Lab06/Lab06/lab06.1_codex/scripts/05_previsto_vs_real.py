"""Nome compatível com o enunciado; encaminha para o gráfico existente."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("05_grafico_previsto_vs_real.py")), run_name="__main__")
