# dynamic-tdma-sim

Simulator and experiments behind the report *Dynamic TDMA for Wireless Sensor
Networks: Principles, Critical Analysis and a Simulation Study* (Vasilis Kladis,
Department of Electrical and Computer Engineering, University of Patras). The
compiled report is [`Dynamic_TDMA_Vasilis_Kladis.pdf`](Dynamic_TDMA_Vasilis_Kladis.pdf).

The simulator models convergecast in a wireless sensor network. Its schedulers
address one question, when a dynamic TDMA schedule improves on a static one:

- the dynamic TDMA of Benrebbouh and Louail [1] with its predictors BSPS and BAPS,
  and its random static and random dynamic baselines;
- a static schedule ordered along the routing tree (Tree-ST);
- the level thresholds of Chittapragada et al. [3];
- two predictors with bounded memory, D-BAPS and EWMA;
- an oracle that knows which nodes hold readings.

Duty-cycled CSMA is included only as a reference outside TDMA.

## Layout

```
wsn/                              simulator package
  topology.py                     unit-disk networks, minimum-hop routing tree, conflict rules A, B and T
  sim.py                          TDMA simulation, traffic models, single-leaf slot traces
  csma.py                         duty-cycled CSMA
experiments/                      scripts that write results/ and figures/
tests/                            checks against the worked examples and closed forms of the report
results/                          CSV files used in the report
figures/                          figures drawn from results/
Dynamic_TDMA_Vasilis_Kladis.pdf   the compiled report
```

## Setup

```
pip install -r requirements.txt
python -m pytest
```

## Reproducing the results

```
python experiments/section6.py
python experiments/section8.py
python experiments/plots_section6.py
python experiments/plots_section8.py
```

The experiment scripts take a list of parts (see `--help`), `--workers` and `--out`;
the plot scripts take `--results` and `--out`. Every random draw is seeded, so a
re-run reproduces `results/` exactly.

## Model

- Networks: nodes placed uniformly in a square sized for density sigma, radio range
  50 m, sink at the centre, minimum-hop routing tree. Only the sink's connected
  component is simulated.
- Time: 15 ms slots. One TDMA frame starts every round of 100 slots (1.5 s) and the
  radios sleep for the rest of the round; `back_to_back=True` runs frames back to back.
- Energy: per-slot charges measured on the GINA mote [4].
- Traffic: Bernoulli, on/off with geometric run lengths, and on/off with
  intermittent readings. Readings are merged at every hop.

## References

1. C. Benrebbouh and L. Louail, "Dynamic TDMA for wireless sensor networks," Proc. WOCC, 2021.
2. I. A. Bouchedjera and L. Louail, "Latency and energy efficient routing-aware TDMA for
   wireless sensor networks," Int. J. Informatics and Applied Mathematics, 2020.
3. B. M. Chittapragada, R. Kumar, V. Doley and B. R. Chandavarkar, "Improving dynamic TDMA
   for wireless sensor networks," Proc. ICCCNT, 2023.
4. X. Vilajosana et al., "A realistic energy consumption model for TSCH networks,"
   IEEE Sensors Journal, vol. 14, no. 2, 2014.

## License

MIT — see [LICENSE](LICENSE).
