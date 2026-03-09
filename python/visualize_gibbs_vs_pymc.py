from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIBBS_REPORT = PROJECT_ROOT / "artifacts" / "comprehensive-ml-workflow-report-gibbs.json"
PYMC_REPORT = PROJECT_ROOT / "artifacts" / "comprehensive-ml-workflow-report.json"
PLOT_PATH = PROJECT_ROOT / "docs" / "images" / "gibbs-vs-pymc-loso.png"
TABLE_PATH = PROJECT_ROOT / "docs" / "gibbs-vs-pymc-loso.md"


def extract_metrics(payload: dict) -> dict[str, float]:
    final = payload["calibratedVsRawMetrics"]["bayesianCalibratedResidualEnsembleLOSO"]
    intervals = payload["calibratedVsRawMetrics"]["intervalMetrics"]
    return {
        "purity_rmse": float(final["purity"]["rmse"]),
        "purity_r2": float(final["purity"]["r2"]),
        "yield_rmse": float(final["yield"]["rmse"]),
        "yield_r2": float(final["yield"]["r2"]),
        "purity_acc": float(final["purity"]["accuracyWithinTolerance"]),
        "yield_acc": float(final["yield"]["accuracyWithinTolerance"]),
        "joint_acc": float(final["combined"]["jointToleranceAccuracy"]),
        "purity_cov": float(intervals["purity"]["coverage"]),
        "yield_cov": float(intervals["yield"]["coverage"]),
        "purity_width": float(intervals["purity"]["meanWidth"]),
        "yield_width": float(intervals["yield"]["meanWidth"]),
    }


def render_table(gibbs: dict[str, float], pymc: dict[str, float]) -> str:
    rows = [
        ("Purity RMSE (↓)", "purity_rmse"),
        ("Purity R² (↑)", "purity_r2"),
        ("Yield RMSE (↓)", "yield_rmse"),
        ("Yield R² (↑)", "yield_r2"),
        ("Purity ±5% Acc (↑)", "purity_acc"),
        ("Yield ±10% Acc (↑)", "yield_acc"),
        ("Joint Acc (↑)", "joint_acc"),
        ("Purity 95% Coverage (≈)", "purity_cov"),
        ("Yield 95% Coverage (≈)", "yield_cov"),
        ("Purity Interval Width (↓)", "purity_width"),
        ("Yield Interval Width (↓)", "yield_width"),
    ]
    lines = [
        "# Gibbs vs PyMC (LOSO) 对照",
        "",
        "| 指标 | Gibbs | PyMC/NUTS | Δ(PyMC-Gibbs) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in rows:
        gv = gibbs[key]
        pv = pymc[key]
        lines.append(f"| {label} | {gv:.4f} | {pv:.4f} | {pv - gv:+.4f} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_plot(gibbs: dict[str, float], pymc: dict[str, float], output: Path) -> None:
    labels = ["Purity RMSE", "Yield RMSE", "Purity R²", "Yield R²", "Joint Acc"]
    g = np.array([gibbs["purity_rmse"], gibbs["yield_rmse"], gibbs["purity_r2"], gibbs["yield_r2"], gibbs["joint_acc"]])
    p = np.array([pymc["purity_rmse"], pymc["yield_rmse"], pymc["purity_r2"], pymc["yield_r2"], pymc["joint_acc"]])
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, g, width, label="Gibbs", color="#4c78a8")
    axes[0].bar(x + width / 2, p, width, label="PyMC/NUTS", color="#f58518")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_title("LOSO Core Metrics")
    axes[0].axhline(0.0, color="#666", linewidth=0.8)
    axes[0].legend()

    labels2 = ["Purity Cov", "Yield Cov", "Purity Width", "Yield Width"]
    g2 = np.array([gibbs["purity_cov"], gibbs["yield_cov"], gibbs["purity_width"], gibbs["yield_width"]])
    p2 = np.array([pymc["purity_cov"], pymc["yield_cov"], pymc["purity_width"], pymc["yield_width"]])
    x2 = np.arange(len(labels2))
    axes[1].bar(x2 - width / 2, g2, width, label="Gibbs", color="#4c78a8")
    axes[1].bar(x2 + width / 2, p2, width, label="PyMC/NUTS", color="#f58518")
    axes[1].set_xticks(x2, labels2, rotation=20, ha="right")
    axes[1].set_title("Interval Coverage and Width")
    axes[1].legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    gibbs_report = json.loads(GIBBS_REPORT.read_text(encoding="utf-8"))
    pymc_report = json.loads(PYMC_REPORT.read_text(encoding="utf-8"))
    gibbs_metrics = extract_metrics(gibbs_report)
    pymc_metrics = extract_metrics(pymc_report)

    table = render_table(gibbs_metrics, pymc_metrics)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table, encoding="utf-8")

    render_plot(gibbs_metrics, pymc_metrics, PLOT_PATH)
    print(f"Wrote table: {TABLE_PATH}")
    print(f"Wrote plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
