"""Figures of Section VI, drawn from the CSV files in results/.

Each panel has axis labels and at most one short legend; captions in the report carry
the explanation. Every scheme keeps its colour and marker in all figures, so the
figures stay readable in grayscale.

Usage: python experiments/plots_section6.py [--results DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

STYLE = {
    "bsps":    dict(color="#2a78d6", marker="o", label="BSPS"),
    "baps":    dict(color="#eb6834", marker="s", label="BAPS"),
    "dbaps":   dict(color="#eb6834", marker="p", label="D-BAPS", ls=(0, (4, 2))),
    "ewma":    dict(color="#1baf7a", marker="^", label="EWMA"),
    "lt":      dict(color="#eda100", marker="D", label="LT"),
    "rand_st": dict(color="#e87ba4", marker="P", label="Rand-ST"),
    "rand_dt": dict(color="#008300", marker="X", label="Rand-DT"),
    "static":  dict(color="#4a3aa7", marker="h", label="Tree-ST"),
    "oracle":  dict(color="#52514e", marker="v", label="Oracle"),
}
INK, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": "#0b0b0b",
    "xtick.color": INK, "ytick.color": INK, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220,
})
W, H = 3.4, 2.2


def axes():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def line(ax, x, y, key, **kw):
    st = STYLE[key]
    opts = dict(color=st["color"], marker=st["marker"], label=st["label"], lw=1.2,
                ms=3.8, mec="white", mew=0.5, zorder=3, ls=st.get("ls", "-"))
    opts.update(kw)
    if opts["lw"] == 0:
        opts["ls"] = "none"
    ax.plot(x, y, **opts)


def legend(ax, **kw):
    opts = dict(frameon=False, handlelength=1.6, handletextpad=0.4,
                columnspacing=0.9, labelspacing=0.3, borderaxespad=0.2)
    opts.update(kw)
    ax.legend(**opts)


def save(fig, name, tight=True):
    """Write PDF and PNG; legend strips keep their fixed width with tight=False."""
    FIG.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=0.3)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("  ", name)


def mean_ci(df, by, col):
    g = df.groupby(by)[col]
    m, s, n = g.mean(), g.std(), g.count()
    return m, 1.96 * s / np.sqrt(n)


def fig_ceiling(c):
    order = ["bsps", "baps", "ewma", "lt", "oracle"]
    fig, ax = axes()
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    d = c[c.p < 1.0]
    for k in order:
        m, _ = mean_ci(d[d.scheme == k], "p", "removal")
        line(ax, m.index, m.values, k)
    ax.set(xlabel="Offered load $p$", ylabel="Idle slots removed", xlim=(-0.02, 0.92),
           ylim=(-0.02, 1.06), yticks=[0, 0.25, 0.5, 0.75, 1.0])
    legend(ax, ncol=2, loc="center right", bbox_to_anchor=(1.0, 0.66))
    save(fig, "fig_ceiling_removal")
    fig, ax = axes()
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    d = c[c.p > 0.0]
    for k in order:
        m, _ = mean_ci(d[d.scheme == k], "p", "delayed")
        line(ax, m.index, m.values, k)
    ax.set(xlabel="Offered load $p$", ylabel="Readings delayed", xlim=(0, 1.02),
           ylim=(-0.02, 0.6), yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    save(fig, "fig_ceiling_delayed")


def fig_trap(t):
    fig, ax = axes()
    f0 = int(t.switch.iloc[0])
    ax.axvline(f0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    offsets = {"bsps": 0, "baps": 6, "dbaps": 12, "ewma": 18, "lt": 24, "oracle": 30}
    for k in ["bsps", "baps", "dbaps", "ewma", "lt", "oracle"]:
        d = t[t.policy == k].sort_values("frame")
        y = d.granted.rolling(10, min_periods=1).mean()
        line(ax, d.frame, y, k, lw=1.1, markevery=(offsets[k], 36))
    ax.set(xlabel="Frame", ylabel="Frames with a slot (10-frame mean)",
           xlim=(0, 400), ylim=(-0.03, 1.06), yticks=[0, 0.25, 0.5, 0.75, 1.0])
    legend(ax, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    save(fig, "fig_trap")


def fig_release(r):
    fig, ax = axes()
    x = np.array([0, 210])
    ax.plot(x, x, color=STYLE["baps"]["color"], lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.plot(x, 0.75 * x, color=STYLE["baps"]["color"], lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.plot(x, (1 / 0.6 - 1) * x, color=STYLE["lt"]["color"], lw=0.8, ls=(0, (4, 3)), zorder=2)
    det = r[r.p1 == 1.0]
    for k in ["baps", "lt", "dbaps"]:
        d = det[det.policy == k].sort_values("f0")
        line(ax, d.f0, d.delay, k, lw=0, ms=4.2)
    # BSPS (1 frame) and EWMA (2 frames) almost coincide, so their markers alternate
    for k, start in (("bsps", 0), ("ewma", 1)):
        d = det[det.policy == k].sort_values("f0").iloc[start::2]
        line(ax, d.f0, d.delay, k, lw=0, ms=4.2)
    d = r[(r.policy == "baps") & (r.p1 == 0.8)].sort_values("f0")
    line(ax, d.f0, d.delay, "baps", lw=0, ms=4.2, mfc="white",
         mec=STYLE["baps"]["color"], mew=0.9, label="BAPS, random")
    ax.set(xlabel="Frames spent busy, $f_0$", ylabel="Frames until release",
           xlim=(0, 208), ylim=(-6, 215))
    legend(ax, loc="upper left", ncol=1)
    save(fig, "fig_release")


def fig_latency(c, b):
    fig, ax = axes()
    d = pd.concat([c, b])
    d = d[d.p > 0.0]
    for k in ["rand_dt", "rand_st", "static", "bsps", "baps", "oracle"]:
        m, _ = mean_ci(d[d.scheme == k], "p", "latency")
        line(ax, m.index, m.values, k)
    ax.set(xlabel="Offered load $p$", ylabel="Latency (slots)", xlim=(0, 1.02),
           ylim=(0, 135), yticks=range(0, 121, 20))
    legend(ax, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 0.73))
    save(fig, "fig_latency_load")


def fig_cost(c, k):
    fig, ax = axes()
    for key in ["static", "oracle"]:
        m, _ = mean_ci(c[c.scheme == key], "p", "energy_uC")
        line(ax, m.index, m.values, key)
    m, _ = mean_ci(c[c.scheme == "bsps"], "p", "energy_uC")
    line(ax, m.index, m.values, "bsps", label="BSPS, free updates")
    m, _ = mean_ci(k[k.overhead == "ack"], "p", "energy_uC")
    line(ax, m.index, m.values, "bsps", ls=(0, (4, 2)), mfc="white",
         mec=STYLE["bsps"]["color"], mew=0.8, label="BSPS, ACK updates")
    m, _ = mean_ci(k[k.overhead == "flood"], "p", "energy_uC")
    line(ax, m.index, m.values, "bsps", ls=(0, (1, 1.5)), marker="x", mec=STYLE["bsps"]["color"],
         mew=0.9, label="BSPS, broadcast updates")
    ax.set(xlabel="Offered load $p$", ylabel="Charge per node per round (µC)",
           xlim=(-0.02, 1.02), ylim=(480, 720))
    legend(ax, loc="lower right", ncol=1)
    save(fig, "fig_cost_energy")


def main():
    global FIG
    ap = argparse.ArgumentParser(description="Figures of Section VI.")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    FIG = args.out
    res = args.results
    c = pd.read_csv(res / "ceiling.csv")
    b = pd.read_csv(res / "baselines.csv")
    k = pd.read_csv(res / "cost.csv")
    fig_ceiling(c)
    fig_trap(pd.read_csv(res / "trap.csv"))
    fig_release(pd.read_csv(res / "release.csv"))
    fig_latency(c, b)
    fig_cost(c, k)
    d = pd.concat([c, b])
    for col in ("removal", "delayed", "latency", "energy_uC"):
        _, ci = mean_ci(d[d.p.between(0.01, 0.99)], ["scheme", "p"], col)
        print(f"   max 95% CI half-width {col}: {ci.max():.3f}")


if __name__ == "__main__":
    main()
