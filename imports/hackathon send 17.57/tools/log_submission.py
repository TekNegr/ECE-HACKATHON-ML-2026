"""Ajoute une ligne de soumission Kaggle dans LOG.md (obligatoire à chaque soumission) et tient le budget de 15.

Le gabarit impose d'écrire la ligne AVANT de voir le score : on logue d'abord, on soumet, puis on remplit le score.

Usage :
  python tools/log_submission.py --who "AB7K,Q2M9" --change "..." --why "..." --cv 0.7123
  python tools/log_submission.py --set-score 0.7050      # remplit le score public de la dernière ligne
  python tools/log_submission.py --status
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "LOG.md"
START, END = "<!-- SUBS:START -->", "<!-- SUBS:END -->"
BUDGET = 15

text = LOG.read_text(encoding="utf-8")
block = text.split(START, 1)[1].split(END, 1)[0]
rows = [l for l in block.splitlines() if re.match(r"^\|\s*\d{1,2}[:h]\d{2}\s*\|", l)]

ap = argparse.ArgumentParser()
ap.add_argument("--who", default="")
ap.add_argument("--set-score")
ap.add_argument("--change")
ap.add_argument("--why")
ap.add_argument("--cv", default="")
ap.add_argument("--score", default="")
ap.add_argument("--status", action="store_true")
a = ap.parse_args()

if a.set_score:
    if not rows:
        raise SystemExit("Aucune ligne à compléter.")
    last = rows[-1]
    cells = last.split("|")
    if cells[-2].strip():
        raise SystemExit(f"La dernière ligne a déjà un score : {last}")
    cells[-2] = f" {a.set_score} "
    LOG.write_text(text.replace(last, "|".join(cells), 1), encoding="utf-8")
    print("|".join(cells))
    raise SystemExit(0)
if a.status or not a.change:
    print(f"{len(rows)}/{BUDGET} soumissions utilisées, {BUDGET - len(rows)} restantes")
    for r in rows[-3:]:
        print(r)
    raise SystemExit(0)
if len(rows) >= BUDGET:
    raise SystemExit("Budget de 15 soumissions atteint : rien ajouté.")
for field in ("change", "why"):
    if not getattr(a, field).strip():
        raise SystemExit(f"--{field} est obligatoire (règle LOG.md)")

clean = lambda s: str(s).replace("|", "/").replace("\n", " ").strip()
line = (f"| {datetime.now():%H:%M} | {clean(a.who)} | {clean(a.change)} | {clean(a.why)} | "
        f"{clean(a.cv)} | {clean(a.score)} |")
LOG.write_text(text.replace(END, line + "\n" + END, 1), encoding="utf-8")
print(line)
print(f"{len(rows) + 1}/{BUDGET} soumissions utilisées")
