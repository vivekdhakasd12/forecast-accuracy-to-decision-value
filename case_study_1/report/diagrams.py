"""Schematic diagrams for the report.

Drawn rather than photographed: the pipeline figure has to show where the
leakage gate sits and where the disk boundaries are, and a screenshot of a
directory tree shows neither.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
ORANGE, INK, SLATE, HAIR = "#E64415", "#1A1A1A", "#555555", "#C9C6C0"
SKY, SAND = "#1B6E77", "#F0EDE7"

plt.rcParams.update({"font.size": 8.2, "savefig.dpi": 300, "savefig.bbox": "tight"})


def _box(ax, x, y, w, h, label, sub=None, fc="white", ec=HAIR, lw=1.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha="center", va="center",
            fontsize=8.4, color=INK, zorder=3,
            fontweight="bold" if bold else "normal")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.030, sub, ha="center", va="center",
                fontsize=6.9, color=SLATE, zorder=3, style="italic")


def _arrow(ax, p, q, color=SLATE, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9,
                                 color=color, lw=0.9, linestyle=ls,
                                 shrinkA=1.5, shrinkB=1.5, zorder=1))


def pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(6.9, 5.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.02, 0.975, "SOURCE", fontsize=6.8, color=ORANGE, fontweight="bold")
    _box(ax, 0.06, 0.885, 0.88, 0.072,
         "Open Power System Data  ·  hourly German series, 2016–2020",
         "PV generation · load · price · installed capacity", fc=SAND)

    ax.text(0.02, 0.845, "PREPARE", fontsize=6.8, color=ORANGE, fontweight="bold")
    _box(ax, 0.06, 0.745, 0.41, 0.078, "Quality control",
         "micro-gaps only; longest valid span")
    _box(ax, 0.53, 0.745, 0.41, 0.078, "Capacity normalisation",
         "MW → capacity factor (39.7→50.5 GW trend)")
    _arrow(ax, (0.50, 0.885), (0.265, 0.823))
    _arrow(ax, (0.50, 0.885), (0.735, 0.823))

    _box(ax, 0.06, 0.625, 0.41, 0.078, "Solar geometry (analytic)",
         "NOAA position · Haurwitz & Ineichen clear sky")
    _box(ax, 0.53, 0.625, 0.41, 0.078, "Gated design matrix",
         "nothing may post-date the gate", ec=ORANGE, lw=1.5)
    _arrow(ax, (0.265, 0.745), (0.265, 0.703))
    _arrow(ax, (0.735, 0.745), (0.735, 0.703))
    _arrow(ax, (0.47, 0.664), (0.53, 0.664), color=ORANGE)

    ax.text(0.500, 0.724, "leakage boundary", fontsize=6.2, color=ORANGE,
            ha="center", va="center", style="italic")

    ax.text(0.02, 0.585, "LEARN", fontsize=6.8, color=ORANGE, fontweight="bold")
    _box(ax, 0.06, 0.475, 0.60, 0.086, "Model zoo — one shared contract",
         "persistence · climatology · ridge · GBM · LSTM")
    _box(ax, 0.70, 0.475, 0.24, 0.086, "Conformal", "split CQR · ACI", fc=SAND)
    _arrow(ax, (0.735, 0.625), (0.36, 0.561))
    _arrow(ax, (0.66, 0.518), (0.70, 0.518))

    ax.text(0.02, 0.435, "EVALUATE", fontsize=6.8, color=ORANGE, fontweight="bold")
    _box(ax, 0.06, 0.325, 0.88, 0.078, "Rolling-origin backtest — 3 expanding folds",
         "train   |   90-day calibration slice (never scored)   |   test")
    _arrow(ax, (0.36, 0.475), (0.36, 0.403))
    _arrow(ax, (0.82, 0.475), (0.82, 0.403))

    ax.text(0.02, 0.285, "DECIDE  ·  MONITOR  ·  EXPLAIN", fontsize=6.8, color=ORANGE,
            fontweight="bold")
    _box(ax, 0.06, 0.155, 0.27, 0.105, "Dispatch LP", "two-stage stochastic\nHiGHS · CVaR",
         ec=SKY, lw=1.3)
    _box(ax, 0.365, 0.155, 0.27, 0.105, "Drift & retraining",
         "ADWIN · Page-Hinkley\nbudget-matched", ec=SKY, lw=1.3)
    _box(ax, 0.67, 0.155, 0.27, 0.105, "Explainability",
         "TreeSHAP · ablation\nerror decomposition", ec=SKY, lw=1.3)
    for x in (0.195, 0.50, 0.805):
        _arrow(ax, (x, 0.325), (x, 0.260))

    _box(ax, 0.06, 0.035, 0.88, 0.068, "Results  —  score tables, figures, EUR/day regret",
         "seeded · 22 tests · reproducible via  make all", fc=SAND)
    for x in (0.195, 0.50, 0.805):
        _arrow(ax, (x, 0.155), (x, 0.103))

    p = OUT / "pipeline_architecture.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def protocol() -> Path:
    """The day-ahead gate, drawn to scale on a 48-hour axis."""
    fig, ax = plt.subplots(figsize=(6.9, 1.85))
    ax.set_xlim(-13, 37); ax.set_ylim(0, 1); ax.axis("off")

    ax.annotate("", xy=(37, 0.52), xytext=(-13, 0.52),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.0))
    for h, lab in [(-12, "day D\n00:00"), (0, "gate\n12:00"), (12, "day D+1\n00:00"),
                   (35, "day D+1\n23:00")]:
        ax.plot([h, h], [0.47, 0.57], color=SLATE, lw=0.9)
        ax.text(h, 0.34, lab, ha="center", va="top", fontsize=6.8, color=SLATE)

    ax.add_patch(FancyBboxPatch((-13, 0.60), 13, 0.16,
                 boxstyle="round,pad=0.004,rounding_size=0.01",
                 fc=SAND, ec=HAIR, lw=0.8))
    ax.text(-6.5, 0.68, "observations usable", ha="center", va="center", fontsize=7.2, color=INK)

    ax.add_patch(FancyBboxPatch((12, 0.60), 23, 0.16,
                 boxstyle="round,pad=0.004,rounding_size=0.01",
                 fc="white", ec=SKY, lw=1.2))
    ax.text(23.5, 0.68, "24 hourly positions committed", ha="center", va="center",
            fontsize=7.2, color=SKY)

    ax.plot([0, 0], [0.30, 0.86], color=ORANGE, lw=1.8)
    ax.text(0, 0.93, "GATE CLOSURE", ha="center", fontsize=7.0, color=ORANGE, fontweight="bold")

    ax.annotate("", xy=(12, 0.20), xytext=(0, 0.20),
                arrowprops=dict(arrowstyle="<|-|>", color=SLATE, lw=0.8))
    ax.text(6, 0.12, "12 h", ha="center", fontsize=6.6, color=SLATE)
    ax.annotate("", xy=(35, 0.05), xytext=(0, 0.05),
                arrowprops=dict(arrowstyle="<|-|>", color=SLATE, lw=0.8))
    ax.text(17.5, -0.02, "35 h — the longest horizon scored", ha="center", fontsize=6.6, color=SLATE)

    p = OUT / "day_ahead_protocol.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


if __name__ == "__main__":
    print(pipeline())
    print(protocol())
