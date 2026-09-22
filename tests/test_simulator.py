"""The simulator must reproduce the hand-worked results and closed forms of the report."""
import math

import numpy as np

from wsn import (Bernoulli, Network, OnOff, Params, assign_first_fit, assign_tree,
                 example_network, leaf_trace, random_network, simulate, simulate_csma)
from wsn.sim import IDLE, RX, SLEEP, TX


def idx(*nodes):
    """Node k of the report is index k-1."""
    return [k - 1 for k in nodes]


def formula_latency(net, slot_of, L):
    """Latency of eqs. (3)-(4) per node, with slots counted from 1."""
    s = {v: slot_of[v] + 1 for v in slot_of}
    out = {}
    for v in range(1, net.N):
        dt, cur = s[v], v
        while net.parent[cur] != 0:
            nxt = net.parent[cur]
            dt += (s[nxt] - s[cur]) if s[cur] < s[nxt] else (L - s[cur] + s[nxt]) % L
            cur = nxt
        out[v] = dt
    return out


def test_routing_tree():
    net = example_network()
    assert net.parent == [-1, 0, 0, 1, 1, 2, 5]            # P = [-,1,1,2,2,3,6]
    assert [net.depth[v] for v in range(7)] == [0, 1, 1, 2, 2, 2, 3]


def test_table_iii_schedule():
    net = example_network()
    slot_of, L = assign_tree(net.tree_order, net.masks["B"], net.children)
    assert L == 4
    assert {v + 1: slot_of[v] + 1 for v in slot_of} == {2: 3, 3: 4, 4: 1, 5: 2, 6: 2, 7: 1}
    assert net.failed_receptions(slot_of) == 0
    slot_a, L_a = assign_tree(net.tree_order, net.masks["A"], net.children)
    assert L_a == 4 and net.failed_receptions(slot_a) == 0


def test_latency_ordered_vs_reordered():
    net = example_network()
    slot_of, L = assign_tree(net.tree_order, net.masks["B"], net.children)
    lat = formula_latency(net, slot_of, L)
    assert lat[6] == 4                                       # node 7
    assert sum(lat.values()) / 6 == 3.5
    # reordered schedule of Section IV-C: node 3 in slot 1, nodes 4 and 6 in slot 2,
    # node 2 in slot 3, nodes 5 and 7 in slot 4
    rnd = {v: s - 1 for v, s in {2: 1, 3: 2, 5: 2, 1: 3, 4: 4, 6: 4}.items()}
    assert net.failed_receptions(rnd) == 0
    lat_r = formula_latency(net, rnd, 4)
    assert lat_r[6] == 9
    assert abs(sum(lat_r.values()) / 6 - 28 / 6) < 1e-12


def test_simulated_latency_matches_formula():
    net = example_network()
    res = simulate(net, Params(policy="static", order="tree", rule="B"),
                   Bernoulli(1.0), frames=40, warmup=5, rng=np.random.default_rng(0))
    s = res.summary()
    assert s["frame_len"] == 4 and s["failed"] == 0
    assert abs(s["latency"] - 3.5) < 1e-12


def test_hidden_terminal_detected():
    net = example_network()
    n3, n4 = idx(3, 4)
    assert net.failed_receptions({n3: 0, n4: 0}) == 1        # collision at node 2
    assert not (net.masks["T"][n3] >> n4) & 1                # the literal rule allows it
    assert (net.masks["B"][n3] >> n4) & 1                    # Rule B forbids it


def test_rules_a_and_b_never_collide():
    rng = np.random.default_rng(1)
    for _ in range(20):
        net, _ = random_network(100, 50.0, 12.0, rng)
        for rule in ("A", "B"):
            slot_of, _ = assign_tree(net.tree_order, net.masks[rule], net.children)
            assert net.failed_receptions(slot_of) == 0


def test_bsps_idle_leaf_alternates():
    g = leaf_trace("bsps", [0] * 200)
    assert g[:3] == [True, True, True]
    assert all(g[f] != g[f + 1] for f in range(3, 199))
    assert abs(sum(g[3:]) / len(g[3:]) - 0.5) < 0.01


def test_baps_half_rate_service_is_permanent():
    g = leaf_trace("baps", [0] * 100 + [1] * 400)
    after = g[100:]
    assert abs(sum(after) / len(after) - 0.5) < 0.01
    assert all(after[i] != after[i + 1] for i in range(len(after) - 1))


def test_baps_release_time_equals_busy_time():
    for f0 in (20, 60, 120):
        g = leaf_trace("baps", [1] * f0 + [0] * (3 * f0))
        first_denial = next(f for f in range(f0, len(g)) if not g[f])
        assert abs((first_denial - f0) - f0) <= 2            # Proposition 2, p1 = 1, p2 = 0


def test_reprobe_period_k():
    for k in (2, 3, 4, 6):
        g = leaf_trace("bsps", [0] * 600, k=k)
        assert abs(sum(g[3:]) / len(g[3:]) - 1 / k) < 0.01   # removal 1 - 1/k


def test_round_model_latency():
    net = example_network()
    res = simulate(net, Params(policy="static", order="tree", rule="B"),
                   Bernoulli(1.0), frames=30, warmup=5, rng=np.random.default_rng(0))
    s = res.summary()
    assert s["latency"] == 3.5 and s["pdr"] == 1.0


def test_back_to_back_latency():
    net = example_network()
    res = simulate(net, Params(policy="static", order="tree", rule="B", back_to_back=True),
                   Bernoulli(1.0), frames=30, warmup=5, rng=np.random.default_rng(0))
    assert res.summary()["latency"] == 3.5


def test_csma_sanity():
    net = example_network()
    light = simulate_csma(net, Bernoulli(0.05), rounds=400, warmup=20,
                          rng=np.random.default_rng(2), window=4).summary()
    assert light["pdr"] > 0.95 and light["latency"] < 10
    tdma = simulate(net, Params(policy="static", order="tree", rule="B"), Bernoulli(0.05),
                    frames=400, warmup=20, rng=np.random.default_rng(2)).summary()
    assert light["energy_uC"] > tdma["energy_uC"]
    heavy = simulate_csma(net, Bernoulli(1.0), rounds=200, warmup=20,
                          rng=np.random.default_rng(3), window=4).summary()
    assert 0.0 < heavy["pdr"] <= 1.0


def test_onoff_traffic():
    rng = np.random.default_rng(4)
    for burst in (1.25, 10):
        t = OnOff(0.2, burst)
        x = np.array([t(f, 1000, rng) for f in range(2000)])
        assert abs(x.mean() - 0.2) < 0.01
        c = np.corrcoef(x[:-1].ravel(), x[1:].ravel())[0, 1]
        assert abs(c - (1 - 1 / burst - 0.2 / (burst * 0.8))) < 0.02   # lag-1 correlation


def test_onoff_intermittent_activity():
    rng = np.random.default_rng(6)
    t = OnOff(0.2, 20, q=0.7)
    x = np.array([t(f, 1000, rng) for f in range(3000)])
    assert abs(x.mean() - 0.2) < 0.01
    assert abs(t.pi_on - 0.2 / 0.7) < 1e-12


def test_delayed_readings_match_markov_chain():
    """Star network under BSPS: a leaf is denied exactly after a granted frame it did
    not use, so P(denied) = 1 / (p(2-p)/(1-p) + 2), which tends to 1/2 as p -> 0."""
    net = Network([set(range(1, 6))] + [{0} for _ in range(5)])
    for p in (0.02, 0.2, 0.5):
        s = simulate(net, Params(policy="bsps"), Bernoulli(p), frames=20000, warmup=50,
                     rng=np.random.default_rng(11)).summary()
        expect = 1 / (p * (2 - p) / (1 - p) + 2)
        assert abs(s["delayed"] - expect) < 0.02, (p, s["delayed"], expect)
    for policy in ("static", "oracle"):
        s = simulate(net, Params(policy=policy), Bernoulli(0.2), frames=500, warmup=10,
                     rng=np.random.default_rng(1)).summary()
        assert s["delayed"] == 0.0


def test_dbaps_limits_are_bsps_and_baps():
    """lam = 0 reproduces BSPS and lam = 1 reproduces BAPS exactly."""
    net, _ = random_network(100, 50.0, 10.0, np.random.default_rng(10_003))
    for lam, ref in ((0.0, "bsps"), (1.0, "baps")):
        for p in (0.1, 0.4):
            a = simulate(net, Params(policy="dbaps", lam=lam), Bernoulli(p), 120, 10,
                         np.random.default_rng(5))
            b = simulate(net, Params(policy=ref), Bernoulli(p), 120, 10, np.random.default_rng(5))
            assert a.latencies == b.latencies and a.frame_len == b.frame_len
            assert a.energy == b.energy
    arr = (np.random.default_rng(8).random(3000) < 0.3).tolist()
    assert leaf_trace("dbaps", arr, lam=0.0) == leaf_trace("bsps", arr)
    assert leaf_trace("dbaps", arr, lam=1.0) == leaf_trace("baps", arr)


def test_dbaps_closed_forms():
    """Proposition 4: after a long busy period a leaf keeps its slot for
    1 + floor(ln 2 / ln(1/lam)) frames; after a long inactive period it is served in
    every second frame until it has used ceil(ln((1 - lam)/2) / (2 ln lam)) slots."""
    for lam in (0.8, 0.9, 0.95):
        f0 = 500
        g = leaf_trace("dbaps", [1] * f0 + [0] * 200, lam=lam)
        first_denied = next(f for f in range(f0, len(g)) if not g[f])
        assert first_denied - f0 == 1 + math.floor(math.log(2) / math.log(1 / lam))
        g = leaf_trace("dbaps", [0] * f0 + [1] * 800, lam=lam)
        used = 0
        for f in range(f0, len(g) - 1):
            used += g[f]
            if g[f] and g[f + 1]:
                break
        assert used == math.ceil(math.log((1 - lam) / 2) / (2 * math.log(lam)))
        assert all(g[f + 1:])


def test_random_order_latency_matches_closed_form():
    """Rand-ST at p = 1: the simulated latency equals eqs. (3)-(4) on the same
    schedule, with the frame length replaced by the round length."""
    for i in range(3):
        net, _ = random_network(100, 50.0, 10.0, np.random.default_rng(10_000 + i))
        sim_lat = simulate(net, Params(policy="static", order="random", rule="B"), Bernoulli(1.0),
                           20, 5, np.random.default_rng(20_000 + i)).summary()["latency"]
        rng = np.random.default_rng(20_000 + i)             # replay frame-0 traffic, then the order
        rng.random(net.N)
        nodes = list(range(1, net.N))
        perm = [nodes[int(j)] for j in rng.permutation(len(nodes))]
        slot_of, _ = assign_first_fit(perm, net.masks["B"])
        s = {v: slot_of[v] + 1 for v in slot_of}
        tot = 0
        for v in range(1, net.N):
            t, u = s[v], v
            while net.parent[u] != 0:
                par = net.parent[u]
                t += (s[par] - s[u]) if s[u] < s[par] else (100 - s[u] + s[par])
                u = par
            tot += t
        assert abs(sim_lat - tot / (net.N - 1)) < 1e-9


def test_charge_identities():
    """Without traffic a static schedule charges only the idle listening of parents;
    the oracle at p = 1 charges one transmission and one reception per node."""
    net, _ = random_network(100, 50.0, 10.0, np.random.default_rng(10_001))
    n1 = sum(1 for v in range(1, net.N) if net.parent[v] == 0)
    s = simulate(net, Params(policy="static"), Bernoulli(0.0), 5, 2, np.random.default_rng(0)).summary()
    assert abs(s["energy_uC"] - (100 * SLEEP + (net.N - 1 - n1) * (IDLE - SLEEP) / (net.N - 1))) < 1e-9
    s = simulate(net, Params(policy="oracle"), Bernoulli(1.0), 5, 2, np.random.default_rng(0)).summary()
    expect = 100 * SLEEP + (TX - SLEEP) + (net.N - 1 - n1) * (RX - SLEEP) / (net.N - 1)
    assert abs(s["energy_uC"] - expect) < 1e-9


def test_ack_updates_are_free_when_every_node_sends():
    """With ACK-carried updates only nodes that did not send pay for the schedule,
    so at p = 1 BSPS costs exactly as much as Tree-ST."""
    net, _ = random_network(100, 50.0, 10.0, np.random.default_rng(10_002))
    a = simulate(net, Params(policy="bsps", overhead="ack"), Bernoulli(1.0), 20, 5,
                 np.random.default_rng(3)).summary()
    b = simulate(net, Params(policy="static"), Bernoulli(1.0), 20, 5, np.random.default_rng(3)).summary()
    assert abs(a["energy_uC"] - b["energy_uC"]) < 1e-9
