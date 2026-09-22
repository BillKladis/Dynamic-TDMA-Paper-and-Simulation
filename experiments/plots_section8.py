"""Figures and the summary table (Table IX) of Section VIII, from results/*.csv.

Usage: python experiments/plots_section8.py [--results DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plots_section6 as p6  # noqa: E402
from plots_section6 import GRID, MUTED, STYLE, axes, legend, line, mean_ci, save  # noqa: E402

STYLE["csma"] = dict(color="#e34948", marker="<", label="CSMA")
MAIN = ["static", "bsps", "baps", "oracle", "csma"]
NODES = [25, 50, 100, 150, 200]
W3, H3 = 2.3, 1.72                          # one third of the page width


def panel(w=W3, h=H3):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def series(ax, d, x, col, keys, **kw):
    for k in keys:
        m, _ = mean_ci(d[d.scheme == k], x, col)
        line(ax, m.index, m.values, k, **kw)


def legend_strip(name, handles, **kw):
    fig = plt.figure(figsize=(7.0, 0.22))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               handletextpad=0.4, **kw)
    save(fig, name, tight=False)


def handle(key, **kw):
    opts = dict(color=STYLE[key]["color"], marker=STYLE[key]["marker"], lw=1.2,
                ms=3.8, mec="white", mew=0.5, label=STYLE[key]["label"])
    opts.update(kw)
    return plt.Line2D([], [], **opts)


def fig_density(d):
    fig, ax = axes()
    b = d[d.scheme == "static"].groupby("sigma")["delta_T"].mean()
    ax.plot(b.index, b.values, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    m, _ = mean_ci(d[d.scheme == "rand_st_A"], "sigma", "frame_len")
    line(ax, m.index, m.values, "rand_st", ls=(0, (4, 2)), mfc="white",
         mec=STYLE["rand_st"]["color"], mew=0.8, label="Rand-ST, Rule A")
    series(ax, d, "sigma", "frame_len", ["rand_st", "static", "bsps", "baps", "oracle"])
    ax.set(xlabel="Density $\\sigma$", ylabel="Frame length (slots)", xlim=(5.5, 20.5),
           ylim=(0, 34), xticks=range(6, 21, 2))
    legend(ax, ncol=2, loc="upper left")
    save(fig, "fig_density_frame")
    fig, ax = axes()
    series(ax, d, "sigma", "latency", ["rand_st", "static", "bsps", "baps", "oracle", "csma"])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set(xlabel="Density $\\sigma$", ylabel="Latency (slots)", xlim=(5.5, 20.5),
           ylim=(5, 420), xticks=range(6, 21, 2))
    legend(ax, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig_density_latency")


def fig_csma(s, c, k):
    """Latency against the number of nodes; latency and charge against the offered load."""
    load = pd.concat([c[c.scheme.isin(MAIN)], k])
    load = load[load.p > 0]
    fig, ax = panel()
    series(ax, s, "n", "latency", MAIN)
    ax.set(xlabel="Number of nodes", ylabel="Latency (slots)", ylim=(0, 65), xlim=(10, 215),
           xticks=NODES)
    save(fig, "fig_size_latency")
    xopt = dict(xlim=(0, 1.03), xticks=[0, 0.25, 0.5, 0.75, 1])
    fig, ax = panel()
    series(ax, load, "p", "latency", MAIN)
    ax.set(xlabel="Offered load $p$", ylabel="Latency (slots)", ylim=(0, 145), **xopt)
    save(fig, "fig_load_latency")
    fig, ax = panel()
    series(ax, load, "p", "energy_uC", ["static", "csma"])
    ax.set(xlabel="Offered load $p$", ylabel="Charge per round (µC)", ylim=(0, 1800), **xopt)
    save(fig, "fig_load_energy")
    legend_strip("fig_csma_legend", [handle(kk) for kk in MAIN], handlelength=1.8, columnspacing=1.6)


def fig_bursty(b):
    bursts = sorted(b.burst.unique())
    xopt = dict(xscale="log", xlim=(1.1, 60))

    def log_x(ax):
        ax.xaxis.set_major_locator(FixedLocator(bursts))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{x:g}" for x in bursts])

    fig, ax = panel()
    series(ax, b, "burst", "latency", ["static", "bsps", "baps", "lt", "ewma", "oracle", "csma"])
    ax.set(xlabel="Mean active run (rounds)", ylabel="Latency (slots)", ylim=(0, None), **xopt)
    log_x(ax)
    save(fig, "fig_bursty_latency")
    fig, ax = panel()
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    series(ax, b, "burst", "delayed", ["bsps", "baps", "lt", "ewma"])
    ax.set(xlabel="Mean active run (rounds)", ylabel="Readings delayed", ylim=(-0.02, 0.6),
           yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], **xopt)
    log_x(ax)
    save(fig, "fig_bursty_delayed")
    fig, ax = panel()
    series(ax, b, "burst", "energy_uC", ["static", "bsps", "baps", "ewma", "oracle"])
    m, _ = mean_ci(b[b.scheme == "bsps_ack"], "burst", "energy_uC")
    line(ax, m.index, m.values, "bsps", ls=(0, (4, 2)), mfc="white",
         mec=STYLE["bsps"]["color"], mew=0.8, label="BSPS, ACK updates")
    ax.set(xlabel="Mean active run (rounds)", ylabel="Charge per round (µC)", **xopt)
    log_x(ax)
    save(fig, "fig_bursty_energy")
    handles = [handle(kk) for kk in ["static", "bsps", "baps", "lt", "ewma", "oracle", "csma"]]
    handles.append(handle("bsps", ls=(0, (4, 2)), mfc="white", mec=STYLE["bsps"]["color"], mew=0.8,
                          label="BSPS, ACK updates"))
    legend_strip("fig_bursty_legend", handles, handlelength=1.8, columnspacing=1.2)


def fig_memory(m):
    """Latency against the memory 1/(1 - lam); memory 1 is BSPS, the last D-BAPS point BAPS."""
    inf = 400.0
    ticks = [1, 2, 5, 10, 20, 100, inf]
    for traffic in ("memoryless", "steady", "intermittent"):
        d = m[m.traffic == traffic]
        fig, ax = panel()
        ax.axhline(d[d.scheme == "static"].latency.mean(), color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
        for scheme in ("dbaps", "ewma"):
            x = d[d.scheme == scheme].copy()
            x["mem"] = [inf if lam >= 1 else 1 / (1 - lam) for lam in x.lam]
            g = x.groupby("mem").latency.mean()
            line(ax, g.index, g.values, scheme)
        ax.set(xscale="log", xlim=(0.8, 550), ylim=(0, 62),
               xlabel="Memory (frames)", ylabel="Latency (slots)")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["1", "2", "5", "10", "20", "100", "∞"])
        save(fig, f"fig_memory_{traffic}")
    handles = [handle(k, ls=STYLE[k].get("ls", "-")) for k in ("dbaps", "ewma")]
    handles.append(plt.Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="Tree-ST"))
    legend_strip("fig_memory_legend", handles, handlelength=2.2, columnspacing=1.6)


def summary_table(res):
    """Table IX: every scheme at the reference point on the networks of Section VI.
    The first five rows add the design elements of the protocol one at a time."""
    d = pd.concat([pd.read_csv(res / f) for f in ("ceiling.csv", "baselines.csv",
                                                   "csma_load.csv", "reference.csv")])
    d = d[(d.p - 0.2).abs() < 1e-9]
    rows = ["rand_st_A", "rand_st", "static", "bsps", "oracle",
            "rand_dt", "baps", "dbaps", "lt", "ewma", "csma"]
    out = []
    for key in rows:
        x = d[d.scheme == key]
        out.append({"scheme": key, "frame_len": x.frame_len.mean(), "latency": x.latency.mean(),
                    "latency_p95": x.latency_p95.mean(), "delayed": x.delayed.mean(),
                    "energy_uC": x.energy_uC.mean(), "pdr": x.pdr.mean(), "networks": len(x)})
    tab = pd.DataFrame(out)
    tab.to_csv(res / "summary_reference.csv", index=False)
    print(tab.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Figures and Table IX of Section VIII.")
    ap.add_argument("--results", type=Path, default=p6.ROOT / "results")
    ap.add_argument("--out", type=Path, default=p6.ROOT / "figures")
    args = ap.parse_args()
    p6.FIG = args.out
    res = args.results
    d = pd.read_csv(res / "density.csv")
    size = pd.read_csv(res / "size.csv")
    csma_load = pd.read_csv(res / "csma_load.csv")
    fig_density(d)
    fig_csma(size, pd.read_csv(res / "ceiling.csv"), csma_load)
    summary_table(res)
    fig_bursty(pd.read_csv(res / "bursty.csv"))
    fig_memory(pd.read_csv(res / "memory.csv"))
    for name, frame, x in (("density", d, "sigma"), ("size", size, "n"), ("csma_load", csma_load, "p")):
        for col in ("frame_len", "latency", "energy_uC", "pdr"):
            _, ci = mean_ci(frame, ["scheme", x], col)
            worst = ci.idxmax() if ci.notna().any() else None
            print(f"   {name:9s} {col:9s} max 95% CI half-width {ci.max():7.3f} at {worst}")


if __name__ == "__main__":
    main()
