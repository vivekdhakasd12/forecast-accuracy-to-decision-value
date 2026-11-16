"""Render the opening card for the final demo video.

The first cut put five minutes of code before any result. That is the wrong order
for an examiner, who wants to know what was found before deciding whether to watch
how it was built. This card goes first and answers that in twenty seconds.

    .venv/bin/python presentation/make_opener.py

Output: presentation/_opener.mp4 (1920x1080, 25 fps, silent)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

OUT = Path(__file__).resolve().parent / "_opener.mp4"
ORANGE, INK, SLATE, SAND, HAIR = "#E64415", "#1A1A1A", "#555555", "#F7F4F0", "#D8D4CE"
FPS = 25

TITLE_HOLD = 5.0          # seconds on the title alone
REVEAL_GAP = 2.6          # seconds between findings appearing
FINAL_HOLD = 5.0          # seconds with all five up

FINDINGS = [
    ("22.5%", "A forecast range that claimed 90% confidence was right 22.5% of the time."),
    ("57%",   "Calibration fixed it, and improved the interval score by 57%."),
    ("r = 0.88", "Better forecasts usually bought cheaper decisions. Usually, not always."),
    ("+20%",  "The most cautious strategy cut the shortfalls and still cost the most."),
    ("7 days", "A brand new site does better borrowing a model than training its own."),
]


def build() -> None:
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")

    title = [
        fig.text(0.5, 0.70, "Probabilistic Day-Ahead Solar Forecasting", fontsize=37,
                 fontweight="bold", ha="center", color=INK),
        fig.text(0.5, 0.615, "and Risk-Aware Battery Dispatch", fontsize=37,
                 fontweight="bold", ha="center", color=INK),
        fig.text(0.5, 0.50, "Case Study 1  ·  Topic 8, AI for Clean Energy", fontsize=17,
                 ha="center", color=ORANGE),
        fig.text(0.5, 0.435, "Devendra Singh Dhakad  ·  Matriculation 100004684", fontsize=15,
                 ha="center", color=SLATE),
        fig.text(0.5, 0.385, "Supervisor: Prof. Dr. Mehrdad Jalali", fontsize=15,
                 ha="center", color=SLATE),
        fig.text(0.5, 0.335, "SRH University of Applied Sciences Heidelberg", fontsize=15,
                 ha="center", color=SLATE),
    ]

    writer = FFMpegWriter(fps=FPS, bitrate=5200)
    with writer.saving(fig, str(OUT), dpi=120):
        for _ in range(int(TITLE_HOLD * FPS)):
            writer.grab_frame()
        for t in title:
            t.remove()

        head = fig.text(0.5, 0.875, "What this project found", fontsize=32,
                        fontweight="bold", ha="center", color=INK)
        rule = fig.add_artist(plt.Line2D([0.36, 0.64], [0.838, 0.838], color=ORANGE, lw=2.5))
        shown = []
        for i, (big, text) in enumerate(FINDINGS):
            y = 0.735 - i * 0.135
            fig.patches.append(plt.Rectangle((0.075, y - 0.048), 0.85, 0.108,
                                             transform=fig.transFigure, facecolor=SAND,
                                             edgecolor="none", zorder=0))
            shown.append(fig.text(0.115, y + 0.005, big, fontsize=27, fontweight="bold",
                                  color=ORANGE, va="center"))
            shown.append(fig.text(0.315, y + 0.005, text, fontsize=19, color=INK,
                                  va="center"))
            for _ in range(int(REVEAL_GAP * FPS)):
                writer.grab_frame()

        foot = fig.text(0.5, 0.055, "How it was built follows.", fontsize=16,
                        ha="center", color=SLATE, style="italic")
        for _ in range(int(FINAL_HOLD * FPS)):
            writer.grab_frame()

    plt.close(fig)
    secs = TITLE_HOLD + len(FINDINGS) * REVEAL_GAP + FINAL_HOLD
    print(f"wrote {OUT.name}  ({secs:.0f}s)")


if __name__ == "__main__":
    build()
