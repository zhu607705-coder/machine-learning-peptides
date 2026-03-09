from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = PROJECT_ROOT / "artifacts" / "comparisons" / "bayes_likelihood_compare.json"
TABLE_PATH = PROJECT_ROOT / "docs" / "bayes-likelihood-ablation.md"
PLOT_PATH = PROJECT_ROOT / "docs" / "images" / "bayes-likelihood-ablation.png"


def build_table(payload: dict) -> str:
    normal = payload["normal"]["metrics"]
    student_t = payload["student_t"]["metrics"]
    rows = [
        ("Purity RMSE (↓)", normal["purity"]["rmse"], student_t["purity"]["rmse"]),
        ("Purity R² (↑)", normal["purity"]["r2"], student_t["purity"]["r2"]),
        ("Yield RMSE (↓)", normal["yield"]["rmse"], student_t["yield"]["rmse"]),
        ("Yield R² (↑)", normal["yield"]["r2"], student_t["yield"]["r2"]),
        ("Joint Accuracy (↑)", normal["combined"]["jointToleranceAccuracy"], student_t["combined"]["jointToleranceAccuracy"]),
    ]
    lines = [
        "# Bayes Likelihood Ablation (normal vs student_t)",
        "",
        f"- Mode: `{payload.get('mode', 'unknown')}`",
        "",
        "| Metric | normal | student_t | Δ(student_t-normal) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, n_val, s_val in rows:
        lines.append(f"| {label} | {n_val:.4f} | {s_val:.4f} | {s_val - n_val:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def build_plot(payload: dict, path: Path) -> None:
    normal = payload["normal"]["metrics"]
    student_t = payload["student_t"]["metrics"]
    labels = ["Purity RMSE", "Yield RMSE", "Purity R²", "Yield R²", "Joint Acc"]
    n_values = np.array(
        [
            normal["purity"]["rmse"],
            normal["yield"]["rmse"],
            normal["purity"]["r2"],
            normal["yield"]["r2"],
            normal["combined"]["jointToleranceAccuracy"],
        ],
        dtype=float,
    )
    s_values = np.array(
        [
            student_t["purity"]["rmse"],
            student_t["yield"]["rmse"],
            student_t["purity"]["r2"],
            student_t["yield"]["r2"],
            student_t["combined"]["jointToleranceAccuracy"],
        ],
        dtype=float,
    )

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, n_values, width=width, label="normal", color="#4c78a8")
    ax.bar(x + width / 2, s_values, width=width, label="student_t", color="#f58518")
    ax.axhline(0.0, color="#666", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title("Bayesian Likelihood Ablation")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    payload = json.loads(COMPARE_PATH.read_text(encoding="utf-8"))
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(build_table(payload), encoding="utf-8")
    build_plot(payload, PLOT_PATH)
    print(f"Wrote table: {TABLE_PATH}")
    print(f"Wrote plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
