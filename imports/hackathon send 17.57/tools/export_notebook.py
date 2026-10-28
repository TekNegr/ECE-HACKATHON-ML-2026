"""Convertit hackathon.py (cellules `# %%`) en hackathon.ipynb, le notebook rendu sur Boostcamp.

Usage : python tools/export_notebook.py [--execute]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "hackathon.py", ROOT / "hackathon.ipynb"


def cells_from_percent(text: str) -> list:
    cells, kind, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif kind == "code" and body.strip():
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


nb = nbformat.v4.new_notebook(cells=cells_from_percent(SRC.read_text(encoding="utf-8")))
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, DST)
print(f"{DST.name} : {len(nb.cells)} cellules")
if "--execute" in sys.argv:
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=1800", str(DST)], check=True, cwd=ROOT)
    print("notebook exécuté (sorties incluses)")
