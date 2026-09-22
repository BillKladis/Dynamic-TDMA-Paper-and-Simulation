"""Network model: unit-disk graph, sink, minimum-hop routing tree and conflict masks.

Nodes are numbered 0..N-1 with the sink at 0. A conflict mask is an int used as a
bitset: bit w of mask[u] is set when u and w may not share a slot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def side_for_density(n: int, r: float, sigma: float) -> float:
    """Side a of the square that gives density sigma = pi r^2 n / a^2."""
    return math.sqrt(math.pi * r * r * n / sigma)


def _bits(nodes) -> int:
    m = 0
    for v in nodes:
        m |= 1 << v
    return m


@dataclass
class Network:
    adj: list[set[int]]                         # radio neighbours
    parent: list[int] = field(init=False)       # parent[0] = -1
    depth: list[int] = field(init=False)
    children: list[list[int]] = field(init=False)
    height: list[int] = field(init=False)       # 0 for leaves
    tree_order: list[int] = field(init=False)   # leaves first, then by decreasing depth
    masks: dict[str, list[int]] = field(init=False)

    def __post_init__(self):
        self.N = len(self.adj)
        self._bfs_tree()
        self._heights()
        self.tree_order = sorted(range(1, self.N),
                                 key=lambda v: (self.height[v], -self.depth[v], v))
        self.leaves = [v for v in range(1, self.N) if not self.children[v]]
        self.max_depth = max(self.depth)
        # largest node degree in the tree: lower bound on an aggregated frame [Incel 2012]
        self.delta_T = max(len(self.children[v]) + (v != 0) for v in range(self.N))
        self.masks = {"A": self._masks_A(), "B": self._masks_B(), "T": self._masks_T()}

    def _bfs_tree(self):
        """Minimum-hop tree, built breadth-first from the sink; ties go to the lowest id."""
        N = self.N
        self.parent = [-1] * N
        self.depth = [-1] * N
        self.depth[0] = 0
        frontier = [0]
        while frontier:
            nxt = []
            for u in frontier:
                for w in sorted(self.adj[u]):
                    if self.depth[w] < 0:
                        self.depth[w] = self.depth[u] + 1
                        self.parent[w] = u
                        nxt.append(w)
            frontier = sorted(nxt)
        if min(self.depth) < 0:
            raise ValueError("network is not connected to the sink")
        self.children = [[] for _ in range(N)]
        for v in range(1, N):
            self.children[self.parent[v]].append(v)

    def _heights(self):
        self.height = [0] * self.N
        for v in sorted(range(self.N), key=lambda x: -self.depth[x]):
            if self.children[v]:
                self.height[v] = 1 + max(self.height[c] for c in self.children[v])

    def _masks_A(self):
        """Rule A: nodes within two hops never share a slot."""
        m = [0] * self.N
        for u in range(1, self.N):
            s = set(self.adj[u])
            for w in self.adj[u]:
                s |= self.adj[w]
            s -= {u, 0}
            m[u] = _bits(s)
        return m

    def _masks_B(self):
        """Rule B (Bouchedjera and Louail): one-hop neighbours conflict, and two-hop
        neighbours conflict when the node between them is the parent of either."""
        m = [0] * self.N
        for u in range(1, self.N):
            s = set(self.adj[u])
            s |= self.adj[self.parent[u]]           # the middle node is u's parent
            for x in self.adj[u]:                   # the middle node x is w's parent
                s.update(self.children[x])
            s -= {u, 0}
            m[u] = _bits(s)
        return m

    def _masks_T(self):
        """Literal reading of Benrebbouh and Louail: conflict within two hops of the tree."""
        m = [0] * self.N
        for u in range(1, self.N):
            p = self.parent[u]
            s = {p} | set(self.children[u])
            if p > 0:
                s.add(self.parent[p])
            s.update(self.children[p])
            for c in self.children[u]:
                s.update(self.children[c])
            s -= {u, 0, -1}
            m[u] = _bits(s)
        return m

    def failed_receptions(self, slot_of: dict[int, int]) -> int:
        """Transmissions that fail under the protocol model: the receiver transmits,
        or another transmitter in the same slot is within range of the receiver."""
        by_slot: dict[int, set[int]] = {}
        for v, s in slot_of.items():
            by_slot.setdefault(s, set()).add(v)
        fails = 0
        for tx in by_slot.values():
            for u in tx:
                r = self.parent[u]
                if r in tx or any(w != u and w in self.adj[r] for w in tx):
                    fails += 1
        return fails


def random_network(n: int, r: float, sigma: float, rng: np.random.Generator):
    """n nodes uniform in a square sized for density sigma, sink at the centre.

    Returns the Network of the sink's connected component and the fraction of
    nodes in it.
    """
    a = side_for_density(n, r, sigma)
    pos = rng.uniform(0.0, a, size=(n, 2))
    pos[0] = (a / 2, a / 2)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    near = d2 <= r * r
    np.fill_diagonal(near, False)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for w in np.nonzero(near[u])[0]:
            w = int(w)
            if w not in seen:
                seen.add(w)
                stack.append(w)
    comp = sorted(seen)                          # the sink keeps index 0
    index = {v: i for i, v in enumerate(comp)}
    adj = [set() for _ in comp]
    for v in comp:
        for w in np.nonzero(near[v])[0]:
            w = int(w)
            if w in index:
                adj[index[v]].add(index[w])
    return Network(adj), len(comp) / n


def example_network() -> Network:
    """The seven-node example of the report; node k of the report is index k-1."""
    edges = [(1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (4, 5), (3, 6), (6, 7)]
    adj = [set() for _ in range(7)]
    for a, b in edges:
        adj[a - 1].add(b - 1)
        adj[b - 1].add(a - 1)
    return Network(adj)
