"""Nome compatível com o enunciado; encaminha para o relatório existente."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("07_relatorio_automatico.py")), run_name="__main__")
