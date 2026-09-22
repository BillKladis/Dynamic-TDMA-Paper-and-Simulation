"""Experiments of Section VI (critical analysis) of the report.

Writes to results/ (or --out):
  trap.csv, release.csv, dbaps_times.csv  single-leaf slot grants (Propositions 2-4)
  ceiling.csv       all predictors against the offered load (Proposition 1)
  baselines.csv     random static and random dynamic baselines
  cost.csv          BSPS with ACK-carried and broadcast schedule updates
  reprobe.csv       BSPS against the re-probe period k
  literal.csv       frame length and failed receptions under Rules A, B and T
  connectivity.csv  share of nodes connected to the sink at low density

Usage: python experiments/section6.py [part ...] [--workers N] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wsn import Bernoulli, Params, assign_tree, leaf_trace, random_network, simulate  # noqa: E402

N_NODES, RANGE_M, SIGMA = 100, 50.0, 10.0
NETWORKS, WARMUP, FRAMES = 50, 50, 250
LOADS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DBAPS_LAM = 0.9


def network(i: int, sigma: float = SIGMA):
    """Network i is the same for every scheme and load (common random numbers)."""
    return random_network(N_NODES, RANGE_M, sigma, np.random.default_rng(10_000 + i))


def write(out: Path, name: str, rows: list[dict]):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name} ({len(rows)} rows)")


def run_pool(fn, jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs, chunksize=4))


def job_load(args):
    i, scheme, p, overhead = args
    net, reach = network(i)
    prm = Params(policy=scheme, order="tree", rule="B", overhead=overhead)
    s = simulate(net, prm, Bernoulli(p), FRAMES, WARMUP, np.random.default_rng(20_000 + i)).summary()
    return {"net": i, "scheme": scheme, "overhead": overhead, "p": p, "nodes": net.N,
            "reachable": reach, **s}


def job_baseline(args):
    i, policy, p = args
    net, reach = network(i)
    prm = Params(policy=policy, order="random", rule="B")
    s = simulate(net, prm, Bernoulli(p), FRAMES, WARMUP, np.random.default_rng(20_000 + i)).summary()
    return {"net": i, "scheme": "rand_st" if policy == "static" else "rand_dt",
            "overhead": "none", "p": p, "nodes": net.N, "reachable": reach, **s}


def job_reprobe(args):
    i, k, p = args
    net, _ = network(i)
    prm = Params(policy="bsps", order="tree", rule="B", k=k)
    s = simulate(net, prm, Bernoulli(p), FRAMES, WARMUP, np.random.default_rng(30_000 + i)).summary()
    return {"net": i, "k": k, "p": p, **s}


def job_literal(args):
    i, sigma = args
    net, _ = network(i, sigma)
    out = {"net": i, "sigma": sigma, "transmissions": net.N - 1}
    for rule in ("A", "B", "T"):
        slot_of, L = assign_tree(net.tree_order, net.masks[rule], net.children)
        out[f"L_{rule}"] = L
        out[f"failed_{rule}"] = net.failed_receptions(slot_of)
    return out


def dbaps_rows():
    """Release and escape times of D-BAPS against lam, simulated and in closed form."""
    rows = []
    for lam in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
        g = leaf_trace("dbaps", [1] * 2000 + [0] * 800, lam=lam)
        release = next(f for f in range(2000, len(g)) if not g[f]) - 2000
        g = leaf_trace("dbaps", [0] * 2000 + [1] * 4000, lam=lam)
        used = 0
        for f in range(2000, len(g) - 1):
            used += g[f]
            if g[f] and g[f + 1]:
                break
        rows.append({"lam": lam, "memory": 1 / (1 - lam), "release_sim": release,
                     "release_formula": 1 + math.floor(math.log(2) / math.log(1 / lam)),
                     "escape_used_sim": used,
                     "escape_used_formula": max(1, math.ceil(math.log((1 - lam) / 2) / (2 * math.log(lam)))),
                     "escape_frames_sim": f + 1 - 2000})
    return rows


def trap_rows():
    f0, horizon = 100, 400
    arrivals = [0] * f0 + [1] * (horizon - f0)
    rows = []
    for pol in ("bsps", "baps", "dbaps", "ewma", "lt", "oracle"):
        for f, g in enumerate(leaf_trace(pol, arrivals, lam=DBAPS_LAM)):
            rows.append({"policy": pol, "frame": f, "switch": f0, "granted": int(g)})
    return rows


def release_rows():
    rows = []
    for f0 in range(10, 201, 10):
        arrivals = [1] * f0 + [0] * (4 * f0 + 50)
        for pol in ("bsps", "baps", "dbaps", "ewma", "lt"):
            g = leaf_trace(pol, arrivals, lam=DBAPS_LAM)
            first = next(f for f in range(f0, len(g)) if not g[f])
            rows.append({"policy": pol, "p1": 1.0, "p2": 0.0, "f0": f0,
                         "seeds": 1, "delay": first - f0})
        # BAPS with random usage: p1 = 0.8 before f0, p2 = 0.1 after
        rng = np.random.default_rng(40_000 + f0)
        delays = []
        for _ in range(400):
            arr = np.concatenate([rng.random(f0) < 0.8, rng.random(6 * f0 + 50) < 0.1])
            g = leaf_trace("baps", arr.tolist())
            first = next((f for f in range(f0, len(g)) if not g[f]), None)
            if first is not None:
                delays.append(first - f0)
        rows.append({"policy": "baps", "p1": 0.8, "p2": 0.1, "f0": f0,
                     "seeds": len(delays), "delay": float(np.mean(delays))})
    return rows


def part_traces(out, workers):
    write(out, "trap.csv", trap_rows())
    write(out, "release.csv", release_rows())
    write(out, "dbaps_times.csv", dbaps_rows())


def part_ceiling(out, workers):
    jobs = [(i, s, p, "none") for s in ("static", "bsps", "baps", "ewma", "lt", "oracle")
            for p in LOADS for i in range(NETWORKS)]
    write(out, "ceiling.csv", run_pool(job_load, jobs, workers))


def part_baselines(out, workers):
    jobs = [(i, pol, p) for pol in ("static", "randdt") for p in LOADS for i in range(NETWORKS)]
    write(out, "baselines.csv", run_pool(job_baseline, jobs, workers))


def part_cost(out, workers):
    jobs = [(i, "bsps", p, oh) for oh in ("ack", "flood") for p in LOADS for i in range(NETWORKS)]
    write(out, "cost.csv", run_pool(job_load, jobs, workers))


def part_reprobe(out, workers):
    jobs = [(i, k, p) for k in (2, 3, 4, 5, 6, 8) for p in (0.0, 0.1) for i in range(NETWORKS)]
    write(out, "reprobe.csv", run_pool(job_reprobe, jobs, workers))


def part_literal(out, workers):
    jobs = [(i, s) for s in (6, 8, 10, 12, 14, 16, 18, 20) for i in range(NETWORKS)]
    write(out, "literal.csv", run_pool(job_literal, jobs, workers))


def part_connectivity(out, workers):
    rows = []
    for sigma in (2, 4, 6, 8, 10):
        for i in range(NETWORKS):
            net, reach = network(i, sigma)
            rows.append({"sigma": sigma, "net": i, "reachable": reach, "nodes": net.N})
    write(out, "connectivity.csv", rows)


PARTS = {"traces": part_traces, "ceiling": part_ceiling, "baselines": part_baselines,
         "cost": part_cost, "reprobe": part_reprobe, "literal": part_literal,
         "connectivity": part_connectivity}


def main():
    ap = argparse.ArgumentParser(description="Experiments of Section VI.")
    ap.add_argument("parts", nargs="*", choices=list(PARTS), help="default: all")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    for name in args.parts or PARTS:
        t0 = time.time()
        print(name)
        PARTS[name](args.out, args.workers)
        print(f"  {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
