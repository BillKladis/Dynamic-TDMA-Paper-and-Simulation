"""Duty-cycled CSMA baseline with an S-MAC-like listen/sleep cycle.

Every round of `round_slots` slots starts with an active window of `window` slots in
which all nodes are awake and contend; the radios sleep for the rest of the round.
Contention is a simplified unslotted IEEE 802.15.4 CSMA-CA:
  * a backoff period (320 us) is far shorter than a 15 ms slot, so a node with a
    fresh packet attempts in the next slot; after a busy channel or a failed attempt
    it waits a random 0..CW-1 slots, with CW doubling from 1 to 8;
  * a node defers when a neighbour started transmitting earlier in the same slot;
    hidden nodes cannot hear each other and may collide;
  * a transmission succeeds when the parent is silent and no other transmitter is
    within range of the parent (protocol model, as for TDMA);
  * after 4 failed attempts (3 retries) the node drops its queued readings.
Senders pay TxDataRxAck, successful receivers RxDataTxAck, nodes that hear a
neighbour transmit pay RxData, and all other awake nodes pay `listen`. Readings are
merged at every hop, as in the TDMA model.
"""
from __future__ import annotations

import numpy as np

from .sim import IDLE, RX, RX_ACK, SLEEP, TX_ACK, Result
from .topology import Network


def simulate_csma(net: Network, traffic, rounds: int, warmup: int, rng: np.random.Generator,
                  window: int, round_slots: int = 100, cw_min: int = 1, cw_max: int = 8,
                  max_retries: int = 3, drain: int = 40, listen: float = IDLE) -> Result:
    """`listen` is the charge of an awake slot without traffic. The default, the TSCH
    Idle slot (radio on for a 2.6 ms guard time), is a lower bound; listening for the
    whole 15 ms slot at 11.6 mA costs about 174 uC."""
    N, parent, adj = net.N, net.parent, net.adj
    nodes = list(range(1, N))
    buf: list[list[int]] = [[] for _ in range(N)]
    cw = [cw_min] * N
    tries = [0] * N
    backoff = [0] * N
    res = Result()
    total = warmup + rounds + drain

    def new_data(v):
        backoff[v] = int(rng.integers(cw[v]))

    for r in range(total):
        T = r * round_slots
        measuring = warmup <= r < warmup + rounds
        arrivals = traffic(r, N, rng)
        for v in nodes:
            if arrivals[v]:
                if not buf[v]:
                    new_data(v)
                buf[v].append(r)
                if measuring:
                    res.generated += 1
        e = [SLEEP * round_slots] * N
        for s in range(window):
            ready = [v for v in nodes if buf[v] and backoff[v] == 0]
            for v in nodes:
                if buf[v] and backoff[v] > 0:
                    backoff[v] -= 1
            # carrier sense: ready nodes start at random offsets within the slot, and a
            # node that hears a neighbour already transmitting backs off again
            start = {v: rng.random() for v in ready}
            tx: set[int] = set()
            for v in sorted(ready, key=start.get):
                if any(w in tx for w in adj[v]):
                    cw[v] = min(2 * cw[v], cw_max)
                    backoff[v] = int(rng.integers(cw[v]))
                else:
                    tx.add(v)
            ok_rx: set[int] = set()
            got_data: list[int] = []
            for u in sorted(tx):
                p = parent[u]
                if p not in tx and not any(w != u and w in adj[p] for w in tx):
                    ok_rx.add(p)
                    payload, buf[u] = buf[u], []
                    cw[u], tries[u] = cw_min, 0
                    if p == 0:
                        done = T + s + 1
                        for g in payload:
                            if warmup <= g < warmup + rounds:
                                res.latencies.append(done - g * round_slots)
                                res.delivered += 1
                    else:
                        if not buf[p]:
                            got_data.append(p)
                        buf[p].extend(payload)
                else:
                    tries[u] += 1
                    cw[u] = min(2 * cw[u], cw_max)
                    if tries[u] > max_retries:
                        res.lost += sum(1 for g in buf[u] if warmup <= g < warmup + rounds)
                        buf[u], tries[u], cw[u] = [], 0, cw_min
                    else:
                        backoff[u] = int(rng.integers(cw[u]))
            for p in got_data:
                new_data(p)
            if measuring:
                for v in nodes:
                    if v in tx:
                        e[v] += TX_ACK - SLEEP
                    elif v in ok_rx:
                        e[v] += RX_ACK - SLEEP
                    elif any(w in tx for w in adj[v]):
                        e[v] += RX - SLEEP
                    else:
                        e[v] += listen - SLEEP
        if measuring:
            res.energy.append(sum(e[v] for v in nodes) / len(nodes))
            res.frame_len.append(window)
            res.radio_on.append(window)
            res.duty_b.append(1.0)
    res.frames = rounds
    return res
