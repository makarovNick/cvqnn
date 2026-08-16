"""
Validate the generated notebook before running it in Colab:
  - JSON validity and nbformat structure;
  - syntax of every code cell (lines with ! or % magics are stripped first);
  - that names used across cells are defined somewhere.

The last check catches the most common notebook bug: a cell relying on a
variable that lived in an earlier draft and never made it into the final one.
"""

import ast
import json
import os
import sys

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "cvqnn_visualization.ipynb")

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

print(f"nbformat {nb['nbformat']}.{nb['nbformat_minor']}, "
      f"cells: {len(nb['cells'])}, accelerator={nb['metadata'].get('accelerator')}")

errors = []
defined = set(dir(__builtins__)) | {"__name__", "__file__", "get_ipython"}
used_before_def = []

code_cells = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

for idx, cell in code_cells:
    src_lines = cell["source"]
    # Colab magics are not valid Python - drop them before parsing
    clean = [ln for ln in src_lines if not ln.lstrip().startswith(("!", "%"))]
    src = "\n".join(clean)

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        errors.append(f"cell {idx}: SyntaxError line {e.lineno}: {e.msg}")
        continue

    # what this cell defines
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                defined.add(a.asname or a.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)

    # what this cell reads
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined:
                used_before_def.append((idx, node.id))

print(f"\n=== syntax: {len(code_cells)} code cells ===")
if errors:
    for e in errors:
        print("  FAIL", e)
else:
    print("  OK - every cell parses")

print("\n=== names used before being defined ===")
if used_before_def:
    seen = {}
    for idx, name in used_before_def:
        seen.setdefault(name, idx)
    for name, idx in sorted(seen.items(), key=lambda kv: kv[1]):
        print(f"  cell {idx}: {name}")
else:
    print("  OK - none")

sys.exit(1 if errors else 0)
