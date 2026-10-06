"""Build a local, read-only gallery for the M0 annotation and group audits."""

from __future__ import annotations

import argparse
import csv
from html import escape
import os
from pathlib import Path
from urllib.parse import quote

from src.posturaai.dataset import resolve_image_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_gallery(root: Path, output: Path) -> dict[str, int | str]:
    root, output = root.resolve(), output.resolve()
    samples_path = root / "outputs/visualizations/ground_truth/sample_list.csv"
    decisions_path = root / "data/srkd/audit_decisions.csv"
    samples, pairs = read_csv(samples_path), read_csv(decisions_path)
    files = [samples_path, decisions_path]

    def url(path: Path) -> str:
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing file or path outside PosturaAI: {path}")
        files.append(path)
        return escape(quote(os.path.relpath(path, output.parent), safe="/"), quote=True)

    def picture(path: Path, label: str) -> str:
        target = url(path)
        return (f'<figure><a href="{target}" target="_blank">'
                f'<img src="{target}" loading="lazy" alt="{escape(label, quote=True)}"></a>'
                f'<figcaption>{escape(label)}</figcaption></figure>')

    sections = [f"""<!doctype html>
<html lang="pt"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PosturaAI — revisão M0</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#18212d}}
main{{max-width:1320px;margin:auto;padding:24px}} a{{color:#145fa6}}
nav{{display:flex;gap:20px;flex-wrap:wrap}} article{{background:white;padding:20px;margin:20px 0;border-radius:12px}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:16px}}
figure{{margin:0}} img{{display:block;width:100%;height:auto}} figcaption{{padding:8px 0}}
.pending{{color:#8a4b00}} code{{overflow-wrap:anywhere}} h2,h3{{scroll-margin-top:16px}}
</style><main>
<h1>PosturaAI — revisão M0</h1>
<p class="pending">Auditoria pendente. Esta galeria apresenta evidência; não aprova anotações ou grupos.</p>
<nav><a href="#annotations">{len(samples)} anotações</a><a href="#groups">{len(pairs)} pares de grupos</a>
<a href="{url(decisions_path)}">CSV de decisões</a></nav>
<p>Abre cada imagem para a ver em tamanho original. Regista as decisões dos pares no CSV:
<code>same</code> para a mesma sequência ou cenário indistinguível, <code>different</code> para grupos distintos.
Preenche também <code>reviewer</code> e documenta dúvidas em <code>note</code>.
A distância visual é apenas um indicador para selecionar exemplos.</p>
<h2 id="annotations">Anotações: lateralidade, skeleton e pés</h2>
<p>Ciano: lado esquerdo; laranja: direito; branco: pontos centrais; verde: ligações provisórias.
Regista a revisão em <code>docs/data/visual_audit.md</code>, incluindo os IDs revistos,
ligações dos pés, pontos fora da imagem e verificação de resize/flip.</p>"""]
    for sample in samples:
        label = f"Imagem {sample['image_id']} — {sample['selection_reason']}"
        sections.append(f"<article id=\"annotation-{int(sample['image_id'])}\"><h3>{escape(label)}</h3>"
                        f"<p>Box fora: {escape(sample['bbox_outside_image'])}; "
                        f"pontos fora: {escape(sample['visible_keypoints_outside_image'])}.</p><div class=\"images\">")
        sections.append(picture(root / sample["source_path"], "Original"))
        sections.append(picture(samples_path.parent / sample["overlay"], "Anotação com índices dos pontos"))
        sections.append("</div></article>")
    sections.append('<h2 id="groups">Pares de grupos</h2>')
    for pair in pairs:
        label = f"{pair['candidate_id']} — {pair['kind']}"
        sections.append(f"<article><h3>{escape(label)}</h3><p>Distância: {escape(pair['distance'])}; "
                        f"ambíguo: {escape(pair['ambiguous'])}; decisão: {escape(pair['decision'] or 'pendente')}."
                        f"</p><div class=\"images\">")
        for side in ("a", "b"):
            image_id = int(pair[f"image_id_{side}"])
            path = resolve_image_path(root, {"file_name": f"{image_id:06d}.jpeg"})
            sections.append(picture(path, f"Imagem {image_id} — grupo {pair[f'group_id_{side}']}"))
        sections.append("</div></article>")
    sections.append("</main></html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return {"annotations": len(samples), "pairs": len(pairs), "checked_files": len(set(files)), "gallery": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations/m0_review.html"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    try:
        summary = build_gallery(args.root, output)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"build_audit_gallery: {exc}\n")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
