"""
Проверка сгенерированного ноутбука до запуска в Colab:
  - валидность JSON и структуры nbformat;
  - синтаксис каждой ячейки кода (строки с ! и % магией отбрасываются);
  - что имена, используемые между ячейками, где-то определены.

Последнее ловит самую частую ошибку в ноутбуках — ячейка опирается на
переменную, которая осталась в черновике и до финальной версии не дожила.
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
      f"ячеек: {len(nb['cells'])}, accelerator={nb['metadata'].get('accelerator')}")

errors = []
defined = set(dir(__builtins__)) | {
    "__name__", "__file__", "get_ipython",
}
used_before_def = []

code_cells = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

for idx, cell in code_cells:
    src_lines = cell["source"]
    # магии Colab не являются валидным Python — убираем их для разбора
    clean = [ln for ln in src_lines
             if not ln.lstrip().startswith(("!", "%"))]
    src = "\n".join(clean)

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        errors.append(f"ячейка {idx}: SyntaxError строка {e.lineno}: {e.msg}")
        continue

    # что ячейка определяет
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
        elif isinstance(node, (ast.comprehension,)):
            pass

    # что ячейка читает
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined:
                used_before_def.append((idx, node.id))

print(f"\n=== синтаксис: {len(code_cells)} ячеек кода ===")
if errors:
    for e in errors:
        print("  FAIL", e)
else:
    print("  OK — все ячейки разбираются")

print("\n=== имена, использованные до определения ===")
if used_before_def:
    seen = {}
    for idx, name in used_before_def:
        seen.setdefault(name, idx)
    for name, idx in sorted(seen.items(), key=lambda kv: kv[1]):
        print(f"  ячейка {idx}: {name}")
else:
    print("  OK — таких нет")

sys.exit(1 if errors else 0)
