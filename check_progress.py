"""Print project progress straight from CONTINUITY.md.

Usage:  python check_progress.py
Counts the [x] / [ ] task boxes in Section 2 of CONTINUITY.md and shows
per-phase and overall completion, plus the next unchecked tasks.
"""
import re
from pathlib import Path

text = Path(__file__).with_name("CONTINUITY.md").read_text(encoding="utf-8")
section = text.split("## 2. Phase checklist", 1)[1].split("## 3.", 1)[0]

phases, current, todo = [], None, []
for line in section.splitlines():
    if line.startswith("### "):
        current = [line[4:].strip(), 0, 0]
        phases.append(current)
    m = re.match(r"- \[( |x|X)\] (.+)", line.strip())
    if m and current:
        current[2] += 1
        if m.group(1).lower() == "x":
            current[1] += 1
        else:
            todo.append(m.group(2))

done = sum(p[1] for p in phases)
total = sum(p[2] for p in phases)
print(f"\nOverall: {done}/{total} tasks ({done / total:.0%})\n")
for name, d, t in phases:
    bar = "#" * d + "." * (t - d)
    print(f"  [{bar:<6}] {d}/{t}  {name}")
print("\nNext tasks:")
for t in todo[:3]:
    print(f"  - {t}")
