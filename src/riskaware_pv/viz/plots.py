"""Figures for the report.

House style: one idea per figure, no chartjunk, colour used only where it
carries information, and every axis labelled with units.  All functions save to
``figures`` and return the path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import FIGURES

PALETTE = {
    "persistence": "#9aa0a6", "climatology": "#c2b280", "climatology_quantiles": "#b0a58f",
    "ridge": "#4c78a8", "gbm_point": "#f58518", "gbm_quantile": "#e45756",
    "gbm_quantile_k": "#72b7b2", "lstm_quantile": "#54a24b",
}
POLICY_COLOR = {"oracle": "#333333", "point": "#f58518", "quantile": "#4c78a8",
                "stochastic": "#54a24b", "cvar": "#e45756"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def _save(fig, name: str) -> Path:
    path = FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _place_labels(fig, ax, points, fontsize: float = 7.5, color: str = "#333333"):
    """Annotate ``points`` = [(text, x, y)] without letting labels overlap.

    Offsets are tried in order and the first that does not collide with an
    already-placed label wins.  Extents come from the real renderer rather than
    from a character-count estimate: the estimate was tried first, put boxes on
    the wrong side of their anchor, and reported collisions that were not there.
    Points in the right-hand part of the range try left-hand offsets first so a
    long label runs back into the plot instead of off the figure.
    """
    if not points:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xs = [x for _, x, _ in points]
    right_edge = min(xs) + 0.70 * ((max(xs) - min(xs)) or 1.0)
    base = [(9, 5), (9, -12), (9, 18), (9, -25), (-9, 5), (-9, -12), (-9, 18), (-9, -25)]
    taken: list = []
    for text, x, y in points:
        cand = [(-a, b) for a, b in base] if x > right_edge else base
        for dx, dy in cand:
            ann = ax.annotate(text, (x, y), fontsize=fontsize, xytext=(dx, dy),
                              textcoords="offset points",
                              ha="left" if dx > 0 else "right", color=color, zorder=6)
            bb = ann.get_window_extent(renderer).expanded(1.06, 1.35)
            if any(bb.overlaps(b) for b in taken):
                ann.remove()
                continue
            taken.append(bb)
            break


def plot_data_overview(frame: pd.DataFrame, X: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    ax = axes[0, 0]
    daily = frame["target"].resample("D").mean()
    ax.plot(daily.index, daily.to_numpy(), lw=0.6, color="#4c78a8")
    ax.set_title("Daily mean PV capacity factor")
    ax.set_ylabel("capacity factor")

    ax = axes[0, 1]
    cap = frame["capacity_mw"].resample("ME").mean() / 1000.0
    ax.plot(cap.index, cap.to_numpy(), lw=1.5, color="#e45756")
    ax.set_title("Installed PV capacity (the reason we normalise)")
    ax.set_ylabel("GW installed")

    ax = axes[1, 0]
    pivot = (frame["target"].groupby([frame.index.month, frame.index.hour]).mean()
             .unstack().to_numpy())
    im = ax.imshow(pivot, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, 23, 1, 12])
    ax.set_title("Mean capacity factor by month and hour (UTC)")
    ax.set_xlabel("hour"); ax.set_ylabel("month")
    fig.colorbar(im, ax=ax, shrink=0.85, label="capacity factor")

    ax = axes[1, 1]
    day = X[X["cos_zenith"] > 1e-3]
    ax.hist(day["k"].dropna(), bins=60, color="#54a24b", alpha=0.85)
    ax.set_title("Clear-sky index $k_t$ (daylight hours)")
    ax.set_xlabel("$k_t = y_t\\,/\\,$clear-sky"); ax.set_ylabel("count")

    fig.suptitle("Dataset overview - German PV, OPSD hourly", fontsize=11, y=1.01)
    fig.tight_layout()
    return _save(fig, "01_data_overview.png")


def plot_leaderboard(summary: pd.DataFrame) -> Path:
    df = summary.reset_index()
    df["label"] = df["model"] + " [" + df["variant"] + "]"
    df = df.sort_values("crps")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(df) + 2.2))

    for ax, col, title in ((axes[0], "crps", "CRPS (lower is better)"),
                           (axes[1], "mae", "MAE (capacity factor)")):
        colors = ["#e45756" if v == "cqr" else "#4c78a8" for v in df["variant"]]
        ax.barh(df["label"], df[col], color=colors, alpha=0.9)
        se = df.get(f"{col}_se")
        if se is not None:
            ax.errorbar(df[col], df["label"], xerr=se.fillna(0), fmt="none",
                        ecolor="#333", elinewidth=0.9, capsize=2.5)
        ax.set_title(title)
        ax.invert_yaxis()
    axes[0].set_xlabel("CRPS"); axes[1].set_xlabel("MAE")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#4c78a8", "#e45756")]
    axes[1].legend(handles, ["raw", "conformalised"], loc="lower right")
    fig.suptitle("Day-ahead forecast leaderboard (daylight hours, fold-averaged)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return _save(fig, "02_leaderboard.png")


def plot_reliability(curves: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1, label="perfect calibration")
    for name, curve in curves.items():
        ls = "--" if "cqr" in name else "-"
        ax.plot(curve["nominal"], curve["observed"], marker="o", ms=4, ls=ls, label=name)
    ax.set_xlabel("nominal quantile level")
    ax.set_ylabel("observed exceedance frequency")
    ax.set_title("Reliability: are the quantiles honest?")
    ax.legend(fontsize=8)
    return _save(fig, "03_reliability.png")


def plot_coverage_vs_width(scores: pd.DataFrame) -> Path:
    """Calibration against sharpness, drawn as movement rather than as points.

    Raw and conformalised results for one model are a *pair*, so plotting them
    as sixteen independent markers both buries the relationship and piles the
    labels on top of one another in the crowded upper-right corner.  Drawing an
    arrow from raw to calibrated shows the effect of calibration directly and
    leaves one label per model instead of two.
    """
    df = scores[scores["subset"] == "daylight"] if "subset" in scores else scores
    agg = df.groupby(["model", "variant"])[["picp90", "pinaw90"]].mean()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    placed: list[tuple[str, float, float]] = []
    ax.axhline(0.90, color="#e45756", ls="--", lw=1.2, zorder=1)
    ax.annotate("nominal 90%", xy=(ax.get_xlim()[0], 0.90), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color="#e45756")

    for model in agg.index.get_level_values(0).unique():
        sub = agg.loc[model]
        raw = sub.loc["raw"] if "raw" in sub.index else None
        cqr = sub.loc["cqr"] if "cqr" in sub.index else None
        color = PALETTE.get(model, "#777777")
        if raw is not None and cqr is not None:
            ax.annotate("", xy=(cqr["pinaw90"], cqr["picp90"]),
                        xytext=(raw["pinaw90"], raw["picp90"]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                        alpha=0.75, shrinkA=3, shrinkB=3), zorder=2)
        if raw is not None:
            ax.scatter(raw["pinaw90"], raw["picp90"], s=46, facecolor="white",
                       edgecolor=color, lw=1.6, zorder=3)
        if cqr is not None:
            ax.scatter(cqr["pinaw90"], cqr["picp90"], s=56, color=color, zorder=3)

        anchor = cqr if cqr is not None else raw
        placed.append((model, float(anchor["pinaw90"]), float(anchor["picp90"])))

    _place_labels(fig, ax, placed)

    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="#555", markersize=7, label="raw"),
        plt.Line2D([], [], marker="o", ls="", color="#555", markersize=7,
                   label="conformalised"),
        plt.Line2D([], [], color="#555", lw=1.3, label="effect of calibration"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_xlabel("PINAW - normalised interval width (sharpness)")
    ax.set_ylabel("PICP - empirical coverage")
    ax.set_title("Calibration vs sharpness: the trade-off conformal prediction resolves")
    ax.margins(x=0.12, y=0.08)
    return _save(fig, "04_coverage_vs_width.png")


def most_variable_window(preds: pd.DataFrame, days: int = 5) -> str:
    """Pick the consecutive stretch with the most day-to-day variation.

    The midpoint of the test set landed on five cloudless June days, where every
    interval has the same width and the figure shows nothing about conditional
    uncertainty.  Selecting on the spread of daily mean output makes the figure
    demonstrate the property being claimed instead of contradicting it.
    """
    daily = preds["y"].groupby(preds.index.normalize()).mean()
    roll = daily.rolling(days).std()
    if roll.notna().sum() == 0:
        return str(preds.index[0].date())
    end = roll.idxmax()
    return str((end - pd.Timedelta(days=days - 1)).date())


def plot_fan_chart(preds: pd.DataFrame, quantiles, start: str, days: int = 5,
                   title: str = "") -> Path:
    q = np.asarray(quantiles, dtype=float)
    sel = preds.loc[start:].head(days * 24)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    n_pairs = len(q) // 2
    for i in range(n_pairs):
        lo, hi = f"q{q[i]:.2f}", f"q{q[-(i + 1)]:.2f}"
        if lo in sel and hi in sel:
            # round(), not int(): 0.95 - 0.05 is 0.8999999999999999 in binary
            # floating point, so truncation labels the 90% band "89%".
            width_pct = int(round((q[-(i + 1)] - q[i]) * 100))
            ax.fill_between(sel.index, sel[lo], sel[hi], color="#4c78a8",
                            alpha=0.18 + 0.16 * i,
                            label=f"{width_pct}% interval")
    ax.plot(sel.index, sel["y"], color="#111", lw=1.5, label="actual")
    ax.plot(sel.index, sel["point"], color="#e45756", lw=1.2, ls="--", label="forecast")
    ax.set_ylabel("capacity factor")
    ax.set_title(title or f"Day-ahead predictive distribution, {days} consecutive days")
    ax.legend(ncol=5, fontsize=8)
    fig.autofmt_xdate()
    return _save(fig, "05_fan_chart.png")


def plot_error_decomposition(by_h: pd.DataFrame, by_hour: pd.DataFrame,
                             by_regime: pd.DataFrame, concentration: pd.Series) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    ax = axes[0, 0]
    ax.plot(by_h.index, by_h["mae"], marker="o", ms=3, color="#4c78a8", label="MAE")
    ax.plot(by_h.index, by_h["bias"], marker="s", ms=3, color="#e45756", label="bias")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("lead time (hours after gate closure)")
    ax.set_ylabel("capacity factor"); ax.set_title("Error by forecast horizon")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(by_hour.index, by_hour["mae"], color="#4c78a8", alpha=0.85)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(by_hour.index, by_hour["mean_y"], color="#f58518", lw=1.6, label="mean output")
    ax.set_xlabel("hour of day (UTC)"); ax.set_ylabel("MAE")
    ax2.set_ylabel("mean capacity factor")
    ax.set_title("Error by hour of day")

    ax = axes[1, 0]
    ax.bar(by_regime.index.astype(str), by_regime["mae"], color="#54a24b", alpha=0.9)
    ax.set_xlabel("cloudiness regime at gate (Q1 = cloudiest)")
    ax.set_ylabel("MAE"); ax.set_title("Error by cloud regime known at forecast time")

    ax = axes[1, 1]
    ax.bar(concentration.index.str.replace("_share", "").str.replace("top_", "top "),
           concentration.to_numpy(), color="#e45756", alpha=0.9)
    ax.set_ylabel("share of total absolute error")
    ax.set_title("Error concentration")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Where the error lives", fontsize=11, y=1.01)
    fig.tight_layout()
    return _save(fig, "06_error_decomposition.png")


def plot_dispatch_results(daily: pd.DataFrame, summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    ax = axes[0]
    order = summary.sort_values("mean_regret_eur").index.tolist()
    ax.bar(order, summary.loc[order, "mean_regret_eur"],
           color=[POLICY_COLOR.get(p, "#777") for p in order], alpha=0.9)
    ax.set_ylabel("mean daily regret vs oracle (EUR)")
    ax.set_title("Cost of imperfect information")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    for policy in order:
        sub = daily[daily["policy"] == policy]["regret"]
        if len(sub):
            ax.hist(sub, bins=45, histtype="step", lw=1.5,
                    color=POLICY_COLOR.get(policy, "#777"), label=policy)
    ax.set_xlabel("daily regret (EUR)"); ax.set_ylabel("days")
    ax.set_title("Regret distribution (the tail is the point)")
    ax.legend(fontsize=8)

    ax = axes[2]
    piv = summary[["mean_imbalance_eur", "mean_day_ahead_eur"]].loc[order]
    bottom = np.zeros(len(order))
    for col, color, lab in (("mean_day_ahead_eur", "#4c78a8", "day-ahead energy"),
                            ("mean_imbalance_eur", "#e45756", "imbalance penalty")):
        ax.bar(order, piv[col], bottom=bottom, color=color, alpha=0.9, label=lab)
        bottom += piv[col].to_numpy()
    ax.set_ylabel("mean daily cost (EUR)"); ax.set_title("Cost composition")
    ax.tick_params(axis="x", rotation=20); ax.legend(fontsize=8)

    fig.suptitle("Downstream dispatch: does a better forecast buy a better decision?",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, "07_dispatch.png")


def plot_forecast_vs_decision(joint: pd.DataFrame) -> Path:
    """The headline figure: forecast quality on one axis, money on the other."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(joint["crps"], joint["mean_regret_eur"], s=90, c="#4c78a8", alpha=0.85,
               zorder=3)
    ax.margins(x=0.14, y=0.12)
    _place_labels(fig, ax, [(r["label"], r["crps"], r["mean_regret_eur"])
                            for _, r in joint.iterrows()], fontsize=7.0)
    if len(joint) > 2:
        z = np.polyfit(joint["crps"], joint["mean_regret_eur"], 1)
        xs = np.linspace(joint["crps"].min(), joint["crps"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="#e45756", ls="--", lw=1.2,
                label=f"fit (r = {joint['crps'].corr(joint['mean_regret_eur']):.2f})")
        ax.legend(fontsize=8)
    ax.set_xlabel("CRPS (forecast quality)")
    ax.set_ylabel("mean daily dispatch regret (EUR)")
    ax.set_title("Forecast skill vs decision value")
    return _save(fig, "08_forecast_vs_decision.png")


def plot_drift(trajectories: dict[str, pd.DataFrame], logs: dict, summary: pd.DataFrame
               ) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1])

    ax = axes[0]
    for policy, traj in trajectories.items():
        ax.plot(traj.index, traj["rolling_mae"], lw=1.6, label=policy)
        for ts in logs[policy].refits[1:]:
            ax.axvline(ts, color="#bbb", lw=0.7, ls=":", zorder=0)
    ax.set_ylabel("30-day rolling MAE (daylight)")
    ax.set_title("Retraining policies at an equal budget")
    ax.legend(fontsize=8)

    ax = axes[1]
    # Plot the *difference* against the static baseline: on a zero-based axis
    # 0.0437 and 0.0442 are visually identical, which would hide the entire
    # result.  Percentage improvement with the refit count attached is what the
    # reader needs to weigh accuracy against retraining spend.
    base = summary.loc["static", "mae"] if "static" in summary.index else summary["mae"].max()
    rel = (100.0 * (base - summary["mae"]) / base).sort_values()
    colors = ["#54a24b" if v > 0 else "#e45756" for v in rel]
    ax.barh(rel.index, rel.to_numpy(), color=colors, alpha=0.9)
    for i, policy in enumerate(rel.index):
        n = int(summary.loc[policy, "n_refits"])
        trig = int(summary.loc[policy, "n_triggers"]) if "n_triggers" in summary else 0
        note = f"  {n} refit{'s' if n != 1 else ''}"
        if policy == "drift":
            note += f" ({trig} trigger{'s' if trig != 1 else ''})"
        ax.text(rel[policy], i, note, va="center", fontsize=8,
                ha="left" if rel[policy] >= 0 else "right")
    ax.axvline(0, color="#666", lw=1.0)
    ax.set_xlabel("MAE improvement vs never retraining (%)")
    ax.set_title("Accuracy gained per retraining budget spent")
    ax.margins(x=0.25)
    fig.tight_layout()
    return _save(fig, "09_drift.png")


def plot_shap(grouped: pd.Series, top: pd.DataFrame, dependence: pd.DataFrame,
              feature_name: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    ax = axes[0]
    ax.barh(grouped.index[::-1], grouped.to_numpy()[::-1] * 100, color="#4c78a8", alpha=0.9)
    ax.set_xlabel("share of total |SHAP| (%)")
    ax.set_title("What the model actually runs on")

    ax = axes[1]
    t = top.head(14).iloc[::-1]
    ax.barh(t["feature"], t["mean_abs_shap"], color="#54a24b", alpha=0.9)
    ax.set_xlabel("mean |SHAP| (capacity factor)")
    ax.set_title("Top individual features")
    ax.tick_params(axis="y", labelsize=7)

    ax = axes[2]
    if len(dependence):
        ax.plot(dependence["bin_center"], dependence["mean_shap"], marker="o", ms=4,
                color="#e45756")
        ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel(feature_name); ax.set_ylabel("mean SHAP contribution")
    ax.set_title(f"Dependence: {feature_name}")

    fig.suptitle("Explaining the day-ahead forecaster (TreeSHAP)", fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, "10_shap.png")


def plot_quantile_sweep(sweep: pd.DataFrame, critical: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(sweep["quantile"], sweep["mean_regret_eur"], marker="o", color="#4c78a8")
    best = sweep.loc[sweep["mean_regret_eur"].idxmin(), "quantile"]
    ax.axvline(best, color="#54a24b", ls="-", lw=1.4, label=f"empirical optimum q={best:.2f}")
    ax.axvline(critical, color="#e45756", ls="--", lw=1.4,
               label=f"newsvendor fractile q={critical:.2f}")
    ax.set_xlabel("commitment quantile of the predictive distribution")
    ax.set_ylabel("mean daily regret (EUR)")
    ax.set_title("Which quantile should you commit to?")
    ax.legend(fontsize=8)
    return _save(fig, "11_quantile_sweep.png")


STRATEGY_STYLE = {
    "local":     dict(color="#4c78a8", marker="o", ls="-",  label="local only"),
    "finetune":  dict(color="#54a24b", marker="s", ls="-",  label="source + local correction"),
    "pooled":    dict(color="#f58518", marker="^", ls="-",  label="pooled, local upweighted"),
    "zero_shot": dict(color="#e45756", marker="",  ls="--", label="zero-shot transfer (no local data)"),
}


def plot_transfer_curves(df: pd.DataFrame) -> Path:
    """MAE against local training budget, one panel per target site.

    The x-axis is logarithmic because the budgets span a factor of nearly 70 and
    the interesting behaviour is all at the short end. Zero-shot is drawn as a
    horizontal line rather than a point: it consumes no local data, so it has a
    value at every budget and its intersection with the local curve is the
    quantity the experiment exists to measure.
    """
    sites = list(dict.fromkeys(df["site"]))
    fig, axes = plt.subplots(1, len(sites), figsize=(4.3 * len(sites), 4.2),
                             sharey=False)
    axes = np.atleast_1d(axes)

    for ax, site in zip(axes, sites):
        sub = df[df["site"] == site]
        name = sub["site_name"].iloc[0]
        lat = sub["latitude"].iloc[0]

        zs = sub[sub["strategy"] == "zero_shot"]["mae"]
        if len(zs):
            ax.axhline(float(zs.iloc[0]), **{k: v for k, v in STRATEGY_STYLE["zero_shot"].items()
                                             if k in ("color", "ls", "label")}, lw=1.6)

        for strategy in ("local", "finetune", "pooled"):
            s = sub[sub["strategy"] == strategy].sort_values("budget_days")
            if not len(s):
                continue
            st = STRATEGY_STYLE[strategy]
            ax.plot(s["budget_days"], s["mae"], color=st["color"], marker=st["marker"],
                    ls=st["ls"], ms=4.5, lw=1.5, label=st["label"])

        # Mark the crossover, using the same definition as the results table.
        loc = sub[sub["strategy"] == "local"].sort_values("budget_days")
        if len(zs) and len(loc):
            from ..transfer import stable_crossover
            cx = stable_crossover(loc["budget_days"], loc["mae"], float(zs.iloc[0]))
            if np.isfinite(cx):
                ax.axvline(cx, color="#888", ls=":", lw=1.2)
                ax.annotate(f"{int(cx)} d", (cx, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(4, -11), textcoords="offset points",
                            fontsize=8, color="#555")

        ax.set_xscale("log")
        ax.set_xticks(sorted(sub["budget_days"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("days of local history")
        ax.set_title(f"{name}  ({lat:.1f}°N)")
    axes[0].set_ylabel("MAE on held-out local test period")
    # Legend below the curves: the upper right is where the crossover
    # annotation sits, and the two collided there.
    axes[-1].legend(fontsize=7.5, loc="center right", framealpha=0.92,
                    frameon=True, edgecolor="none")
    fig.suptitle("A new site with little history: borrow, blend, or train local?",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, "14_transfer_curves.png")


def plot_transfer_summary(cross: pd.DataFrame, df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    x = np.arange(len(cross))
    w = 0.26
    ax.bar(x - w, cross["local_mae_7d"], w, color="#4c78a8", alpha=0.9, label="local, 7 days")
    ax.bar(x, cross["zero_shot_mae"], w, color="#e45756", alpha=0.9, label="zero-shot transfer")
    ax.bar(x + w, cross["pooled_mae_7d"], w, color="#f58518", alpha=0.9,
           label="pooled, 7 days")
    ax.set_xticks(x, cross["site_name"], fontsize=8.5)
    ax.set_ylabel("MAE")
    ax.set_title("What a brand-new site can do in its first week")
    ax.legend(fontsize=8)

    ax = axes[1]
    ok = cross.dropna(subset=["crossover_days"])
    if len(ok):
        ax.bar(ok["site_name"], ok["crossover_days"], color="#4c78a8", alpha=0.9)
        for i, (_, r) in enumerate(ok.iterrows()):
            ax.text(i, r["crossover_days"], f"  {int(r['crossover_days'])} d",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("days of local history")
    ax.set_title("Local history needed to beat borrowing")
    ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Transfer as a small-data strategy", fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, "15_transfer_summary.png")
