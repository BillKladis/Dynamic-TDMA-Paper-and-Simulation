"""Experiments of Section VIII (results) of the report.

Writes to results/ (or --out):
  density.csv      all schemes and CSMA against the density (100 nodes, p = 0.2)
  size.csv         all schemes and CSMA against the number of nodes (sigma = 10, p = 0.2)
  csma_load.csv    CSMA against the offered load
  bursty.csv       all schemes against the mean active run of on/off traffic
  backtoback.csv   latency with frames back to back
  memory.csv       D-BAPS and EWMA against the decay factor, three traffic types
  reference.csv    Rand-ST under Rule A and D-BAPS at the reference point
  rounds.csv       latency and charge against the round length
  qsweep.csv       BSPS and EWMA against the reading probability q within a run

The reference-point experiments use the networks of Section VI. CSMA gets an active
window equal to the Tree-ST frame of the same network, so both use the same channel
time per round.

Usage: python experiments/section8.py [part ...] [--workers N] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from section6 import FRAMES, LOADS, NETWORKS, WARMUP, network, run_pool, write  # noqa: E402
from wsn import (Bernoulli, OnOff, Params, assign_tree, random_network, simulate,  # noqa: E402
                 simulate_csma)

P_REF = 0.2
N_NET = 40                                   # networks per point of the density and size sweeps
SCHEMES = {                                  # None = CSMA
    "rand_st_A": Params(policy="static", order="random", rule="A"),
    "rand_st":   Params(policy="static", order="random", rule="B"),
    "rand_dt":   Params(policy="randdt", order="random", rule="B"),
    "static":    Params(policy="static", order="tree", rule="B"),
    "bsps":      Params(policy="bsps", order="tree", rule="B"),
    "baps":      Params(policy="baps", order="tree", rule="B"),
    "lt":        Params(policy="lt", order="tree", rule="B"),
    "ewma":      Params(policy="ewma", order="tree", rule="B"),
    "oracle":    Params(policy="oracle", order="tree", rule="B"),
    "csma":      None,
}
BURSTS = [1.25, 2, 3, 5, 10, 20, 50]         # mean active run in rounds; 1.25 is Bernoulli(0.2)
BURSTY = {**{k: SCHEMES[k] for k in ("static", "bsps", "baps", "lt", "ewma", "oracle", "csma")},
          "bsps_ack": Params(policy="bsps", order="tree", rule="B", overhead="ack")}
B2B = ("rand_st", "rand_dt", "static", "bsps", "baps", "ewma", "oracle")
LAMBDAS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
MEM_TRAFFIC = {"memoryless": (0.2, 1.25, 1.0), "steady": (0.2, 20, 1.0),
               "intermittent": (0.2, 20, 0.7)}          # (p, mean active run, q)
MEM_WARMUP = 400                             # lets a memory of 100 frames settle
REF_EXTRA = {"rand_st_A": Params(policy="static", order="random", rule="A"),
             "dbaps": Params(policy="dbaps", order="tree", rule="B", lam=0.9)}
ROUNDS = [30, 50, 100, 200, 400]
QS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def tree_frame(net):
    return assign_tree(net.tree_order, net.masks["B"], net.children)[1]


def one(net, scheme, p, seed):
    frame = tree_frame(net)
    rng = np.random.default_rng(seed)
    if SCHEMES[scheme] is None:
        res = simulate_csma(net, Bernoulli(p), FRAMES, WARMUP, rng, window=frame)
    else:
        res = simulate(net, SCHEMES[scheme], Bernoulli(p), FRAMES, WARMUP, rng)
    return {"nodes": net.N, "delta_T": net.delta_T, "depth": net.max_depth,
            "tree_frame": frame, **res.summary()}


def job_density(args):
    i, sigma, scheme = args
    net, reach = random_network(100, 50.0, sigma, np.random.default_rng(50_000 + 97 * i + int(sigma)))
    return {"net": i, "sigma": sigma, "scheme": scheme, "p": P_REF, "reachable": reach,
            **one(net, scheme, P_REF, 60_000 + i)}


def job_size(args):
    i, n, scheme = args
    net, reach = random_network(n, 50.0, 10.0, np.random.default_rng(70_000 + 1013 * i + n))
    return {"net": i, "n": n, "scheme": scheme, "p": P_REF, "reachable": reach,
            **one(net, scheme, P_REF, 80_000 + i)}


def job_csma_load(args):
    i, p = args
    net, reach = network(i)
    return {"net": i, "scheme": "csma", "p": p, "reachable": reach,
            **one(net, "csma", p, 20_000 + i)}


def job_bursty(args):
    i, burst, scheme = args
    net, reach = network(i)
    frame = tree_frame(net)
    rng = np.random.default_rng(90_000 + i)
    traffic = OnOff(P_REF, burst)
    if BURSTY[scheme] is None:
        res = simulate_csma(net, traffic, FRAMES, WARMUP, rng, window=frame)
    else:
        res = simulate(net, BURSTY[scheme], traffic, FRAMES, WARMUP, rng)
    return {"net": i, "burst": burst, "scheme": scheme, "p": P_REF, "reachable": reach,
            "nodes": net.N, "tree_frame": frame, **res.summary()}


def job_b2b(args):
    i, scheme = args
    net, _ = network(i)
    base = SCHEMES[scheme]
    prm = Params(policy=base.policy, order=base.order, rule=base.rule, back_to_back=True)
    s = simulate(net, prm, Bernoulli(P_REF), FRAMES, WARMUP, np.random.default_rng(20_000 + i)).summary()
    return {"net": i, "scheme": scheme, "p": P_REF, "frame_len": s["frame_len"],
            "latency": s["latency"], "latency_p95": s["latency_p95"], "service": s["service"]}


def job_memory(args):
    """D-BAPS (every frame counts) and EWMA (only frames with a slot) with the same decay."""
    i, traffic, scheme, lam = args
    net, _ = network(i)
    p, burst, q = MEM_TRAFFIC[traffic]
    if scheme == "dbaps":
        prm = Params(policy="dbaps", order="tree", rule="B", lam=lam)
    elif scheme == "ewma":
        prm = Params(policy="ewma", order="tree", rule="B", alpha=1.0 - lam)
    else:
        prm = Params(policy=scheme, order="tree", rule="B")
    res = simulate(net, prm, OnOff(p, burst, q), FRAMES, MEM_WARMUP, np.random.default_rng(95_000 + i))
    return {"net": i, "traffic": traffic, "scheme": scheme, "lam": lam, **res.summary()}


def job_reference(args):
    """Schemes of Table IX that Section VI does not run, with the seeds of ceiling.csv."""
    i, scheme = args
    net, reach = network(i)
    res = simulate(net, REF_EXTRA[scheme], Bernoulli(P_REF), FRAMES, WARMUP,
                   np.random.default_rng(20_000 + i))
    return {"net": i, "scheme": scheme, "p": P_REF, "nodes": net.N, "reachable": reach,
            **res.summary()}


def job_rounds(args):
    i, rs, scheme = args
    net, _ = network(i)
    base = SCHEMES[scheme]
    prm = Params(policy=base.policy, order=base.order, rule=base.rule, round_slots=rs)
    res = simulate(net, prm, Bernoulli(P_REF), FRAMES, WARMUP, np.random.default_rng(20_000 + i))
    return {"net": i, "round": rs, "scheme": scheme, **res.summary()}


def job_q(args):
    i, q, scheme = args
    net, _ = network(i)
    prm = (Params(policy="ewma", order="tree", rule="B", alpha=0.3) if scheme == "ewma"
           else Params(policy=scheme, order="tree", rule="B"))
    res = simulate(net, prm, OnOff(P_REF, 20, q), FRAMES, MEM_WARMUP, np.random.default_rng(95_000 + i))
    return {"net": i, "q": q, "scheme": scheme, **res.summary()}


def part_density(out, workers):
    jobs = [(i, s, k) for s in (6, 8, 10, 12, 14, 16, 18, 20) for k in SCHEMES for i in range(N_NET)]
    write(out, "density.csv", run_pool(job_density, jobs, workers))


def part_size(out, workers):
    jobs = [(i, n, k) for n in (25, 50, 100, 150, 200)
            for k in ("rand_st", "rand_dt", "static", "bsps", "baps", "ewma", "oracle", "csma")
            for i in range(N_NET)]
    write(out, "size.csv", run_pool(job_size, jobs, workers))


def part_csma_load(out, workers):
    jobs = [(i, p) for p in LOADS if p > 0 for i in range(NETWORKS)]
    write(out, "csma_load.csv", run_pool(job_csma_load, jobs, workers))


def part_bursty(out, workers):
    jobs = [(i, b, k) for b in BURSTS for k in BURSTY for i in range(NETWORKS)]
    write(out, "bursty.csv", run_pool(job_bursty, jobs, workers))


def part_backtoback(out, workers):
    write(out, "backtoback.csv", run_pool(job_b2b, [(i, k) for k in B2B for i in range(NETWORKS)], workers))


def part_memory(out, workers):
    jobs = [(i, t, s, lam) for t in MEM_TRAFFIC for lam in LAMBDAS for s in ("dbaps", "ewma")
            if not (s == "ewma" and lam == 1.0) for i in range(NETWORKS)]
    jobs += [(i, t, s, -1.0) for t in MEM_TRAFFIC for s in ("static", "oracle") for i in range(NETWORKS)]
    write(out, "memory.csv", run_pool(job_memory, jobs, workers))


def part_reference(out, workers):
    jobs = [(i, k) for k in REF_EXTRA for i in range(NETWORKS)]
    write(out, "reference.csv", run_pool(job_reference, jobs, workers))


def part_rounds(out, workers):
    jobs = [(i, r, s) for r in ROUNDS for s in ("rand_st", "rand_dt", "static", "bsps", "baps", "oracle")
            for i in range(NETWORKS)]
    write(out, "rounds.csv", run_pool(job_rounds, jobs, workers))


def part_qsweep(out, workers):
    jobs = [(i, q, s) for q in QS for s in ("static", "bsps", "ewma") for i in range(NETWORKS)]
    write(out, "qsweep.csv", run_pool(job_q, jobs, workers))


PARTS = {"density": part_density, "size": part_size, "csma_load": part_csma_load,
         "bursty": part_bursty, "backtoback": part_backtoback, "memory": part_memory,
         "reference": part_reference, "rounds": part_rounds, "qsweep": part_qsweep}


def main():
    ap = argparse.ArgumentParser(description="Experiments of Section VIII.")
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
