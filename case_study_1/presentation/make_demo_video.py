"""Render the decision-loop demo video from stored results.

The project has no user interface, so a screen recording would show a terminal.
What it does have is a decision that repeats every day, and that is the thing
worth watching: a distribution arrives, an optimiser turns it into a committed
schedule, reality lands, and somebody pays the difference. This script animates
that loop day by day over the delivery days already settled in experiment 2.

Nothing is recomputed. Every frame is drawn from results/e1_predictions.parquet
and results/e2_daily.csv, so the video cannot disagree with the report.

    .venv/bin/python presentation/make_demo_video.py --days 80

Output: presentation/demo_decision_loop.mp4 (1920x1080, 25 fps)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "case_study_1" / "results"
OUT = ROOT / "case_study_1" / "presentation" / "demo_decision_loop.mp4"

ORANGE, INK, SLATE, HAIR = "#E64415", "#1A1A1A", "#555555", "#D8D4CE"
BLUE, GREEN, RED, SAND = "#4c78a8", "#54a24b", "#e45756", "#F7F4F0"
POLICY_COLOR = {"point": ORANGE, "stochastic": GREEN, "cvar": RED, "quantile": BLUE}
TRACKED = ("point", "stochastic", "cvar", "quantile")

# Captions instead of a voiceover. Keyed to the day index they appear on, and
# written the way a person would say them out loud rather than the way a caption
# generator would write them.
CAPTIONS = [
    (0,  "Here is the decision. Once a day, every day."),
    (5,  "On the left, tomorrow's forecast. Not one number. A whole range of what might happen."),
    (13, "The black line is what actually happened. Watch it arrive, hour by hour."),
    (23, "Then the bill. Same weather, four strategies, four different costs."),
    (35, "At the bottom it all adds up. This is the part that matters."),
    (49, "Green commits to the whole range. Orange commits to a single best guess."),
    (63, "The two cautious strategies did cut the shortfalls. They still paid more for it."),
]

FPS = 25
DRAW_FRAMES = 15      # reality arriving, hour by hour
HOLD_FRAMES = 6       # beat on the settled day
TITLE_SECONDS = 3.5
END_SECONDS = 4.5

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#BBBBBB", "text.color": INK,
    "axes.labelcolor": SLATE, "xtick.color": SLATE, "ytick.color": SLATE,
})


def load(model: str = "ridge", variant: str = "raw"):
    preds = pd.read_parquet(RESULTS / "e1_predictions.parquet")
    preds.index = pd.DatetimeIndex(preds.index)
    preds = preds[(preds["model"] == model) & (preds["variant"] == variant)].sort_index()

    daily = pd.read_csv(RESULTS / "e2_daily.csv")
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    wide = daily.pivot_table(index="date", columns="policy", values="regret")
    cost = daily.pivot_table(index="date", columns="policy", values="total_cost")
    return preds, wide.sort_index(), cost.sort_index()


def day_frame(preds: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    d = preds.loc[(preds.index >= day) & (preds.index < day + pd.Timedelta(days=1))]
    return d.sort_index()


def build(n_days: int, model: str, variant: str) -> None:
    preds, regret, cost = load(model, variant)
    days = [d for d in regret.index if len(day_frame(preds, d)) == 24][:n_days]
    if not days:
        raise SystemExit("no delivery days with a complete forecast")
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    qcols = [f"q{q:.2f}" for q in qs]

    cum = regret.loc[days, list(TRACKED)].cumsum()
    ymax_cum = float(cum.max().max()) * 1.08
    ymax_pv = float(max(day_frame(preds, d)["y"].max() for d in days)) * 1.25

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.5, 1.0, 1.0],
                          left=0.055, right=0.965, top=0.845, bottom=0.155,
                          hspace=0.42, wspace=0.30)
    ax_fan = fig.add_subplot(gs[0, :2])
    ax_bill = fig.add_subplot(gs[0, 2])
    ax_cum = fig.add_subplot(gs[1, :])

    fig.text(0.055, 0.945, "The decision loop, one delivery day at a time",
             fontsize=23, fontweight="bold", color=INK)
    sub = fig.text(0.055, 0.898, "", fontsize=13.5, color=SLATE)
    counter = fig.text(0.965, 0.945, "", fontsize=23, fontweight="bold",
                       color=ORANGE, ha="right")
    counter_lab = fig.text(0.965, 0.900, "", fontsize=11.5, color=SLATE, ha="right")

    caption = fig.text(
        0.5, 0.055, "", fontsize=20.5, color=INK, ha="center", va="center",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=SAND, edgecolor=HAIR, linewidth=1.0),
    )
    caption.set_visible(False)

    def caption_for(day_index: int) -> str:
        active = [t for d, t in CAPTIONS if day_index >= d]
        return active[-1] if active else ""

    writer = FFMpegWriter(fps=FPS, bitrate=5200,
                          metadata={"title": "RiskAware-PV decision loop"})
    total = TITLE_SECONDS * FPS + len(days) * (DRAW_FRAMES + HOLD_FRAMES) + END_SECONDS * FPS
    print(f"rendering {int(total)} frames ({total / FPS:.0f}s) over {len(days)} days")

    with writer.saving(fig, str(OUT), dpi=120):
        # ---- title card ---------------------------------------------------
        for ax in (ax_fan, ax_bill, ax_cum):
            ax.set_visible(False)
        card = [
            fig.text(0.5, 0.60, "Probabilistic Day-Ahead Solar Forecasting", fontsize=34,
                     fontweight="bold", ha="center", color=INK),
            fig.text(0.5, 0.525, "and Risk-Aware Battery Dispatch", fontsize=34,
                     fontweight="bold", ha="center", color=INK),
            fig.text(0.5, 0.435, "Every day: a forecast distribution becomes a committed "
                                 "schedule, then reality arrives and somebody pays.",
                     fontsize=16, ha="center", color=SLATE),
            fig.text(0.5, 0.36, f"{len(days)} settled delivery days, replayed from the "
                                f"stored results", fontsize=14, ha="center", color=ORANGE),
        ]
        sub.set_text(""); counter.set_text(""); counter_lab.set_text("")
        caption.set_visible(False)
        for _ in range(int(TITLE_SECONDS * FPS)):
            writer.grab_frame()
        for t in card:
            t.remove()
        for ax in (ax_fan, ax_bill, ax_cum):
            ax.set_visible(True)

        # ---- the loop -------------------------------------------------------
        for i, day in enumerate(days):
            d = day_frame(preds, day)
            hours = np.arange(24)
            actual = d["y"].to_numpy()
            qv = d[qcols].to_numpy()
            r_today = {p: float(regret.loc[day, p]) for p in TRACKED}
            running = cum.iloc[i]

            for f in range(DRAW_FRAMES + HOLD_FRAMES):
                k = 24 if f >= DRAW_FRAMES else int(round(24 * (f + 1) / DRAW_FRAMES))

                # -- panel 1: the forecast, then reality arriving -------------
                ax_fan.clear()
                for j in range(3):
                    ax_fan.fill_between(hours, qv[:, j], qv[:, -(j + 1)], color=BLUE,
                                        alpha=0.16 + 0.15 * j, lw=0,
                                        label=f"{int(round((qs[-(j+1)] - qs[j]) * 100))}% interval")
                ax_fan.plot(hours, qv[:, 3], color=BLUE, lw=1.8, ls="--",
                            label="forecast median")
                if k:
                    ax_fan.plot(hours[:k], actual[:k], color=INK, lw=2.6, label="what happened")
                    ax_fan.scatter([hours[k - 1]], [actual[k - 1]], s=44, color=ORANGE,
                                   zorder=5)
                ax_fan.set_xlim(-0.5, 23.5); ax_fan.set_ylim(0, ymax_pv)
                ax_fan.set_xlabel("hour of the delivery day (UTC)")
                ax_fan.set_ylabel("solar output (capacity factor)")
                ax_fan.set_title("1.  The forecast is a distribution, not a number",
                                 loc="left", fontsize=14, fontweight="bold", color=INK)
                ax_fan.legend(loc="upper left", fontsize=9.5, ncol=2, framealpha=0.9)

                # -- panel 2: today's bill ------------------------------------
                ax_bill.clear()
                settled = f >= DRAW_FRAMES
                names = list(TRACKED)
                vals = [r_today[p] if settled else 0.0 for p in names]
                ax_bill.barh(names, vals, color=[POLICY_COLOR[p] for p in names], alpha=0.9)
                ax_bill.set_xlim(0, max(1.0, max(r_today.values()) * 1.25))
                ax_bill.invert_yaxis()
                ax_bill.set_xlabel("extra cost today vs perfect foresight (EUR)")
                ax_bill.set_title("2.  Then the bill arrives", loc="left", fontsize=14,
                                  fontweight="bold", color=INK)
                if settled:
                    for y_i, p in enumerate(names):
                        ax_bill.text(r_today[p], y_i, f"  {r_today[p]:,.0f}", va="center",
                                     fontsize=10.5, color=INK)
                else:
                    ax_bill.text(0.5, 0.5, "settling...", transform=ax_bill.transAxes,
                                 ha="center", color=HAIR, fontsize=15, style="italic")

                # -- panel 3: the running score -------------------------------
                ax_cum.clear()
                upto = cum.iloc[: i + 1]
                for p in TRACKED:
                    ax_cum.plot(range(len(upto)), upto[p], color=POLICY_COLOR[p], lw=2.2,
                                label=p)
                    ax_cum.scatter([len(upto) - 1], [upto[p].iloc[-1]], s=30,
                                   color=POLICY_COLOR[p], zorder=5)
                ax_cum.axhline(0, color=SLATE, lw=1.2)
                ax_cum.text(0.4, 0, " perfect foresight", va="bottom", fontsize=10,
                            color=SLATE)
                ax_cum.set_xlim(0, len(days)); ax_cum.set_ylim(-ymax_cum * 0.04, ymax_cum)
                ax_cum.set_xlabel("delivery days settled")
                ax_cum.set_ylabel("cumulative extra cost (EUR)")
                ax_cum.set_title("3.  The gap that accumulates. Lower is better.",
                                 loc="left", fontsize=14, fontweight="bold", color=INK)
                ax_cum.legend(loc="upper left", fontsize=10, ncol=4, framealpha=0.9)

                sub.set_text(f"Delivery day {i + 1} of {len(days)}   ·   "
                             f"{day.date()}   ·   forecaster: {model} [{variant}]")
                gap = running["point"] - running["stochastic"]
                # The label follows the sign: an early run of bad days can put the
                # stochastic policy behind, and "-40 EUR saved" would read as nonsense.
                counter.set_text(f"{abs(gap):,.0f} EUR")
                counter.set_color(ORANGE if gap >= 0 else SLATE)
                counter_lab.set_text(("saved" if gap >= 0 else "lost") +
                                     " so far by committing against the whole distribution")
                caption.set_visible(True)
                caption.set_text(caption_for(i))
                writer.grab_frame()

            if (i + 1) % 20 == 0:
                print(f"  day {i + 1}/{len(days)}")

        # ---- end card --------------------------------------------------------
        fin = cum.iloc[-1]
        for ax in (ax_fan, ax_bill, ax_cum):
            ax.set_visible(False)
        sub.set_text(""); counter.set_text(""); counter_lab.set_text("")
        caption.set_visible(False)
        rows = sorted(TRACKED, key=lambda p: fin[p])
        card = [fig.text(0.5, 0.78, f"After {len(days)} delivery days", fontsize=30,
                         fontweight="bold", ha="center", color=INK)]
        for r, p in enumerate(rows):
            card.append(fig.text(0.34, 0.63 - r * 0.085, p, fontsize=21,
                                 color=POLICY_COLOR[p], fontweight="bold"))
            card.append(fig.text(0.66, 0.63 - r * 0.085, f"{fin[p]:,.0f} EUR", fontsize=21,
                                 color=INK, ha="right"))
        card.append(fig.text(0.5, 0.20,
                             "Committing against the full predictive distribution beat the "
                             "point forecast.\nThe two risk-averse policies cut the "
                             "shortfalls and still cost more.",
                             fontsize=15, ha="center", color=SLATE, linespacing=1.6))
        for _ in range(int(END_SECONDS * FPS)):
            writer.grab_frame()
        for t in card:
            t.remove()

    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=80)
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--variant", default="raw")
    args = ap.parse_args()
    build(args.days, args.model, args.variant)
