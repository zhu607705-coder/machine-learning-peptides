from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "多肽合成机器学习学习报告.md"


def expand_refs(text: str) -> list[int]:
    refs: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            left, right = part.split("-", 1)
            refs.extend(range(int(left), int(right) + 1))
        else:
            refs.append(int(part))
    return refs


def main() -> None:
    content = REPORT.read_text(encoding="utf-8")
    parts = content.split("## 参考文献", 1)
    body = parts[0]
    ref_section = parts[1] if len(parts) > 1 else ""

    cited: set[int] = set()
    for match in re.finditer(r"<sup>\[([^\]]+)\]</sup>", body):
        cited.update(expand_refs(match.group(1)))

    refs = {int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]", ref_section, flags=re.M)}

    missing = sorted(cited - refs)
    unused = sorted(refs - cited)

    print("cited:", sorted(cited))
    print("references:", sorted(refs))
    print("missing_references:", missing)
    print("unused_references:", unused)

    peptimizer_lines = [
        (idx + 1, line)
        for idx, line in enumerate(body.splitlines())
        if "peptimizer" in line.lower() and "<sup>[16]</sup>" not in line and "learningmatter-mit/peptimizer" in line
    ]
    print("peptimizer_lines_without_ref16:", len(peptimizer_lines))
    for lineno, line in peptimizer_lines:
        print(f"  line {lineno}: {line.strip()[:160]}")


if __name__ == "__main__":
    main()
