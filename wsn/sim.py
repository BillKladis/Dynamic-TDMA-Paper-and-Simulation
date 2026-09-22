"""Frame-by-frame simulation of TDMA convergecast schedulers.

A scheduler combines
  rule    conflict rule for sharing a slot: 'A', 'B' or 'T' (see topology.py)
  order   'random' (first fit in a random node order) or 'tree' (leaves first,
          every node after all of its children)
  policy  which nodes receive a slot in a frame:
            static  every node, schedule computed once
            randdt  every node, new random order in every frame
            bsps    used its slot in the previous frame [Benrebbouh 2021]
            baps    used its slot in at least half of all previous frames [Benrebbouh 2021]
            dbaps   BAPS with decaying weights S <- lam S + u, W <- lam W + 1 over all
                    frames, slot if S >= W/2 (lam = 0 gives BSPS, lam = 1 gives BAPS)
            lt      usage ratio over allocated slots >= level threshold [Chittapragada 2023]
            ewma    decayed usage over allocated slots >= theta
            oracle  exactly the nodes that hold readings, and their relays
All predicting policies also grant a slot to a node denied for k-1 consecutive
frames (re-probe, k = 2 in [Benrebbouh 2021]) and to every node with a child that
holds a slot in the frame (relay protection).

Energy uses the charges measured by Vilajosana et al. 2014 (GINA mote, 15 ms slot),
over a collection round of `round_slots` slots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain

import numpy as np

from .topology import Network

# charge per 15 ms slot in microcoulombs [Vilajosana et al. 2014, Table V]
TX, RX, IDLE, SLEEP = 69.6, 72.1, 47.9, 4.9
TX_ACK, RX_ACK = 92.6, 96.3


@dataclass
class Params:
    rule: str = "B"
    order: str = "tree"
    policy: str = "static"
    k: int = 2                  # re-probe period
    bootstrap: int = 3          # static frames before a predicting policy starts
    alpha: float = 0.3          # EWMA weight of the newest frame
    theta: float = 0.5          # EWMA threshold
    k_leaf: float = 0.6         # LT threshold at the deepest level
    k_top: float = 0.3          # LT threshold at depth 1
    lam: float = 0.9            # D-BAPS decay factor
    overhead: str = "none"      # schedule distribution: 'none', 'ack' or 'flood'
    round_slots: int = 100      # slots per collection round
    back_to_back: bool = False  # frames follow each other without rounds


def assign_first_fit(seq, mask):
    occ: list[int] = []
    slot: dict[int, int] = {}
    for v in seq:
        mv, bv = mask[v], 1 << v
        for s, o in enumerate(occ):
            if not o & mv:
                occ[s] = o | bv
                slot[v] = s
                break
        else:
            slot[v] = len(occ)
            occ.append(bv)
    return slot, len(occ)


def assign_tree(seq, mask, children, wrap: bool = False):
    """Slot search along the routing tree.

    A leaf searches from the first slot, any other node from the slot after the
    highest slot of its children, and a slot is added only when none fits. With
    wrap=True the search continues from the first slot, as in Bouchedjera and
    Louail 2020; by default every child stays before its parent.
    """
    occ: list[int] = []
    slot: dict[int, int] = {}
    for v in seq:
        cs = [slot[c] for c in children[v] if c in slot]
        s0 = max(cs) + 1 if cs else 0
        L, mv, bv = len(occ), mask[v], 1 << v
        search = chain(range(s0, L), range(0, min(s0, L))) if wrap else range(s0, L)
        for s in search:
            if not occ[s] & mv:
                occ[s] |= bv
                slot[v] = s
                break
        else:
            slot[v] = L
            occ.append(bv)
    return slot, len(occ)


class Bernoulli:
    """Each node generates a reading at the start of a frame with probability p."""

    def __init__(self, p: float):
        self.p = p

    def __call__(self, f: int, n: int, rng) -> np.ndarray:
        return rng.random(n) < self.p


class OnOff:
    """On/off traffic with long-run load p.

    Every node alternates between active runs, in which it generates a reading in a
    frame with probability q, and inactive runs without readings. Run lengths are
    geometric with mean `burst` frames when active. For q = 1 and burst = 1/(1-p)
    the process is Bernoulli(p).
    """

    def __init__(self, p: float, burst: float, q: float = 1.0):
        self.p, self.burst, self.q = p, burst, q
        on = p / q                                      # stationary share of active nodes
        assert 0 < on < 1, "need p < q"
        self.pi_on = on
        self.leave = 1.0 / burst                        # P(active -> inactive)
        self.enter = self.leave * on / (1.0 - on)       # P(inactive -> active)
        self.on = None

    def __call__(self, f, n, rng):
        if self.on is None:
            self.on = rng.random(n) < self.pi_on
        else:
            u = rng.random(n)
            self.on = np.where(self.on, u >= self.leave, u < self.enter)
        if self.q >= 1.0:
            return self.on.copy()
        return self.on & (rng.random(n) < self.q)


@dataclass
class Result:
    frames: int = 0
    frame_len: list = field(default_factory=list)
    energy: list = field(default_factory=list)       # mean charge per node per round (uC)
    duty_b: list = field(default_factory=list)       # duty cycle as defined in [Benrebbouh 2021]
    radio_on: list = field(default_factory=list)     # mean active slots per node
    latencies: list = field(default_factory=list)    # slots
    wasted: list = field(default_factory=list)       # granted but unused slots
    granted: list = field(default_factory=list)
    idle_leaf: int = 0              # leaf-frames without data
    idle_leaf_removed: int = 0      # ... in which the leaf held no slot
    busy_leaf: int = 0              # leaf-frames with data
    busy_leaf_served: int = 0       # ... in which the leaf held a slot
    failed: int = 0                 # failed receptions
    generated: int = 0              # readings generated in the measured frames
    delayed: int = 0                # ... at a node without a slot in that frame
    delivered: int = 0              # ... that reached the sink
    lost: int = 0                   # ... dropped after the CSMA retry limit

    def summary(self) -> dict:
        def mean(xs):
            return float(np.mean(xs)) if len(xs) else float("nan")

        lat = np.asarray(self.latencies, float)
        return {
            "frame_len": mean(self.frame_len),
            "energy_uC": mean(self.energy),
            "duty_b": mean(self.duty_b),
            "radio_on": mean(self.radio_on),
            "latency": float(lat.mean()) if lat.size else float("nan"),
            "latency_p95": float(np.percentile(lat, 95)) if lat.size else float("nan"),
            "wasted": mean(self.wasted),
            "granted": mean(self.granted),
            "removal": self.idle_leaf_removed / self.idle_leaf if self.idle_leaf else float("nan"),
            "service": self.busy_leaf_served / self.busy_leaf if self.busy_leaf else float("nan"),
            "failed": self.failed,
            "delayed": self.delayed / self.generated if self.generated else float("nan"),
            "pdr": self.delivered / self.generated if self.generated else float("nan"),
        }


def simulate(net: Network, prm: Params, traffic, frames: int, warmup: int,
             rng: np.random.Generator, drain: int = 40) -> Result:
    """Run warmup + frames + drain frames and measure the middle `frames`.

    Each frame starts a collection round of prm.round_slots slots; the radios sleep
    after the frame, and a reading that misses a slot on its path waits for the next
    round. With prm.back_to_back the frames follow each other directly.
    """
    N, parent, children = net.N, net.parent, net.children
    mask = net.masks[prm.rule]
    nodes = list(range(1, N))
    thr_lt = [0.0] * N
    for v in nodes:
        span = max(net.max_depth - 1, 1)
        thr_lt[v] = prm.k_top + (prm.k_leaf - prm.k_top) * (net.depth[v] - 1) / span

    last_used = [0] * N
    denied_run = [0] * N
    hist_sum = [0] * N
    ewma = [1.0] * N
    lt_cnt = [0] * N
    lt_tot = [0] * N
    d_sum = [0.0] * N               # D-BAPS decayed usage; frames without a slot count 0
    d_frames = 0.0                  # D-BAPS decayed frame count, equal for all nodes

    buf: list[list[int]] = [[] for _ in range(N)]   # generation frame of each queued reading
    t_start: list[int] = []
    T = 0
    res = Result()
    static_sched = None
    total = warmup + frames + drain
    dynamic = prm.policy not in ("static", "randdt")

    for f in range(total):
        t_start.append(T)
        measuring = warmup <= f < warmup + frames
        arrivals = traffic(f, N, rng)
        for v in nodes:
            if arrivals[v]:
                buf[v].append(f)
                if measuring:
                    res.generated += 1
        has_data = [bool(buf[v]) for v in range(N)]

        if not dynamic or f < prm.bootstrap:
            granted = set(nodes)
        else:
            granted = set()
            for v in net.tree_order:
                relay = any(c in granted for c in children[v])
                if prm.policy == "oracle":
                    g = has_data[v] or relay
                else:
                    if prm.policy == "bsps":
                        pred = last_used[v] == 1
                    elif prm.policy == "baps":
                        pred = hist_sum[v] >= f / 2
                    elif prm.policy == "dbaps":
                        pred = d_sum[v] >= d_frames / 2
                    elif prm.policy == "ewma":
                        pred = ewma[v] >= prm.theta
                    elif prm.policy == "lt":
                        pred = lt_tot[v] == 0 or lt_cnt[v] / lt_tot[v] >= thr_lt[v]
                    else:
                        raise ValueError(prm.policy)
                    g = relay or pred or denied_run[v] >= prm.k - 1
                if g:
                    granted.add(v)

        if prm.policy == "static" and static_sched is not None:
            slot_of, L = static_sched
        else:
            if prm.order == "random":
                perm = [nodes[int(i)] for i in rng.permutation(len(nodes))]
                seq = [v for v in perm if v in granted]
                slot_of, L = assign_first_fit(seq, mask)
            else:
                seq = [v for v in net.tree_order if v in granted]
                slot_of, L = assign_tree(seq, mask, children)
            if prm.policy == "static":
                static_sched = (slot_of, L)
                res.failed = net.failed_receptions(slot_of)
        if prm.policy != "static" and measuring:
            res.failed += net.failed_receptions(slot_of)
        if measuring:
            res.delayed += sum(1 for v in nodes if arrivals[v] and v not in granted)

        by_slot: list[list[int]] = [[] for _ in range(L)]
        for v, s in slot_of.items():
            by_slot[s].append(v)
        used = {}
        for s in range(L):
            for v in by_slot[s]:
                if buf[v]:
                    used[v] = 1
                    payload, buf[v] = buf[v], []
                    p = parent[v]
                    if p == 0:
                        if f >= warmup:
                            done = T + s + 1
                            for g in payload:
                                if warmup <= g < warmup + frames:
                                    res.latencies.append(done - t_start[g])
                                    res.delivered += 1
                    else:
                        buf[p].extend(payload)
                else:
                    used[v] = 0

        if measuring:
            e = [SLEEP * prm.round_slots] * N
            active = [0] * N
            for v in granted:
                u = used[v]
                if u:
                    e[v] += TX - SLEEP
                    active[v] += 1
                p = parent[v]
                if p:
                    e[p] += (RX if u else IDLE) - SLEEP
                    active[p] += 1
            if dynamic and prm.overhead == "flood":
                for v in nodes:
                    e[v] += RX - SLEEP                  # receive the new schedule
                    if children[v]:
                        e[v] += TX - SLEEP              # and forward it
            elif dynamic and prm.overhead == "ack":
                # Acknowledgments cost every TDMA scheme the same and are not charged.
                # A node that sent learns its next slot from the ACK; others need one reception.
                for v in nodes:
                    if not used.get(v, 0):
                        e[v] += RX - SLEEP
            res.energy.append(sum(e[v] for v in nodes) / len(nodes))
            res.duty_b.append(sum(active[v] for v in nodes) / (len(nodes) * L) if L else 0.0)
            res.radio_on.append(sum(active[v] for v in nodes) / len(nodes))
            res.frame_len.append(L)
            res.granted.append(len(granted))
            res.wasted.append(sum(1 for v in granted if not used[v]))
            for v in net.leaves:
                if has_data[v]:
                    res.busy_leaf += 1
                    res.busy_leaf_served += v in granted
                else:
                    res.idle_leaf += 1
                    res.idle_leaf_removed += v not in granted

        for v in nodes:
            if v in granted:
                u = used[v]
                denied_run[v] = 0
                last_used[v] = u
                hist_sum[v] += u
                ewma[v] = (1 - prm.alpha) * ewma[v] + prm.alpha * u
                lt_cnt[v] += u
                lt_tot[v] += 1
                d_sum[v] = prm.lam * d_sum[v] + u
            else:
                denied_run[v] += 1
                last_used[v] = 0
                d_sum[v] = prm.lam * d_sum[v]
        d_frames = prm.lam * d_frames + 1

        if prm.back_to_back:
            T += max(L, 1)
            continue
        if L > prm.round_slots:
            raise ValueError(f"frame of {L} slots exceeds the round of {prm.round_slots}")
        T += prm.round_slots

    res.frames = frames
    return res


def leaf_trace(policy: str, arrivals, k: int = 2, bootstrap: int = 3,
               alpha: float = 0.3, theta: float = 0.5, K: float = 0.6, lam: float = 0.9):
    """Slot grants of a single leaf under the grant rule of Algorithm 1.

    arrivals[f] is 1 when a reading is generated at the start of frame f; readings
    wait in the queue until the leaf holds a slot. Returns one boolean per frame.
    """
    q = False
    last_used = denied_run = S = cnt = tot = 0
    ew = 1.0
    dS = dW = 0.0
    out = []
    for f, a in enumerate(arrivals):
        q = q or bool(a)
        if f < bootstrap:
            g = True
        elif policy == "oracle":
            g = q
        else:
            if policy == "bsps":
                pred = last_used == 1
            elif policy == "baps":
                pred = S >= f / 2
            elif policy == "dbaps":
                pred = dS >= dW / 2
            elif policy == "ewma":
                pred = ew >= theta
            elif policy == "lt":
                pred = tot == 0 or cnt / tot >= K
            else:
                raise ValueError(policy)
            g = pred or denied_run >= k - 1
        if g:
            u = 1 if q else 0
            q = False
            denied_run = 0
            last_used = u
            S += u
            ew = (1 - alpha) * ew + alpha * u
            cnt += u
            tot += 1
        else:
            u = 0
            denied_run += 1
            last_used = 0
        dS = lam * dS + u
        dW = lam * dW + 1
        out.append(g)
    return out
