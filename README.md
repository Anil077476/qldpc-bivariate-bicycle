# Circuit-level simulation of the [[72,12,6]] bivariate bicycle code

A laptop-scale harness for simulating and decoding the [[72,12,6]] bivariate
bicycle (BB) code under circuit-level depolarizing noise, with optional heralded
erasure. Everything runs on one CPU in Python — no quantum hardware, no cluster.

## What this is, and what it is not

**This is** an independent reproduction of the circuit-level memory benchmark for
a high-rate qLDPC code, together with a measurement of how heralded-erasure
information changes BP convergence, OSD fallback frequency and decoding latency.
It is a learning and reproduction project, written to be checkable.

**This is not** a novel contribution. I initially thought the erasure/latency
measurement here was new. It is not. Before writing anything up I checked the
literature, found that every substantive element already exists, and rewrote
this README to say so. What each part overlaps with is recorded under
[Prior work](#prior-work). The results below are real and reproducible; they are
simply not new.

One thing I have not found reported elsewhere is the DEM duplicate-mechanism
effect in the next section. Even that sits close to existing formal work on
detector-error-model equivalence, so treat it as a practitioner's note rather
than a claim.

---

## A practitioner's note: duplicate error mechanisms

Stim builds a detector error model (DEM) by bundling indistinguishable errors —
same detectors flipped, same effect on the logical observables — into a single
*error mechanism*. That bundling is what makes the DEM a well-conditioned
parity-check matrix for decoding.

When the syndrome-extraction circuit is built by repeating a body:

```python
circuit += (rounds - 1) * body      # emits a REPEAT block
```

the bundling does not happen across the block boundary, and the DEM ends up with
**duplicate columns**: distinct mechanisms with identical signatures.

For the [[72,12,6]] Z-memory at 6 rounds:

| circuit construction | mechanisms | unique signatures | duplicates |
| --- | --- | --- | --- |
| `(rounds-1) * body` (REPEAT) | 3636 | 3096 | **540** |
| fully unrolled | 3096 | 3096 | 0 |

The two circuits are semantically identical — same gates, same noise, same 252
detectors. Only the DEM differs.

Duplicate columns are bad for belief propagation: two identical columns create a
symmetry BP cannot break, so BP stops converging and OSD has to rescue nearly
every shot. The measured logical error rate comes out far worse than the code
actually is:

| p | LER/round, duplicated DEM | LER/round, merged DEM |
| --- | --- | --- |
| 0.0010 | 9.82e-04 | 2.54e-04 |
| 0.0020 | 6.92e-03 | 3.41e-03 |
| 0.0030 | 2.47e-02 | 1.52e-02 |

The fitted sub-threshold slope moves from **2.92** to **3.65**.

This matters because BB codes are known to lean on OSD precisely because of
degeneracy in their Tanner graph. An accidental extra source of degeneracy in
the DEM is easy to introduce and easy to miss.

`merge_duplicate_mechanisms()` in `bb_harness.py` fixes it. The merge is exact,
not an approximation: two independent mechanisms with the same effect and
probabilities `p1`, `p2` behave as one with probability `p1 + p2 - 2*p1*p2`.
It is validated against the unrolled circuit, which Stim bundles correctly on
its own: after merging, both give the same failure count on the same shots.

---

## Files

| file | what it does |
| --- | --- |
| `bb_harness.py` | Code construction, CNOT schedule search, Stim circuit, DEM, duplicate merging, BP-OSD, sweep over `p` |
| `erasure_harness.py` | Adds heralded erasure; decodes every shot twice (herald-blind vs herald-aware) and records BP convergence, iterations and latency |
| `run_parallel.py` | Same experiments split across CPU cores. Identical science, ~7x faster wall clock |
| `plot_figure1.py` | LER vs `p`, log-log, Wilson intervals, fitted slope |
| `plot_figure2.py` | LER and BP convergence vs erasure rate; also flags non-monotonicity |

### Code construction

Built from `l=6, m=6`, `A = x^3 + y + y^2`, `B = y^3 + x + x^2`, giving
`HX = [A|B]`, `HZ = [B^T|A^T]`. Self-verifies `HX·HZ^T = 0 (mod 2)`, weight-6
checks, degree-6 qubits, `k = 12`. `logical_z()` returns 12 independent logical
Z operators as `ker(HX)` modulo `rowspace(HZ)`.

### Schedule search

`find_schedule()` does not assume a CNOT ordering. It searches all 6!x6!
interleaved orderings and keeps those where, for every overlapping
(X-check, Z-check) pair, an even number of shared data qubits have the X-CNOT
before the Z-CNOT — the condition for the detectors to stay deterministic.
Result: 9 distinct overlap patterns, 2736 valid schedules; the first is used.

Published BB-code work generally fixes a schedule from the original Nature
paper. Searching is not a contribution, but it does mean the schedule here is
verified rather than assumed.

### Erasure wiring

`HERALDED_ERASE(pe)` is applied to the data qubits each round, and each herald
measurement becomes its own `DETECTOR`. In the DEM an erasure then appears as

```
error(pe/2) D_herald
error(pe/2) D_herald D_syndrome...
```

so conditioned on the herald firing the damaging mechanism has probability 1/2,
and conditioned on it not firing, 0.

Each shot is decoded twice from the *same* sample:

- **herald-blind** — syndrome only; erasure folded into the average rate
- **herald-aware** — herald rows removed from the matrix (they are observed, not
  inferred) and the erasure priors set per shot to 0.5 or ~0

Both decoders share one sampler, so the comparison is paired.

---

## Install and run

Python 3.10+.

```
python -m venv venv
# Windows:  .\venv\Scripts\Activate.ps1
# Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
```

Baseline sweep:

```
python bb_harness.py --rounds 6 --shots 50000 --ps 0.001 0.0015 0.002 0.003 0.004 --csv baseline.csv
python plot_figure1.py baseline.csv
```

Erasure comparison (parallel; use the plain harness for small smoke tests):

```
python run_parallel.py erasure --rounds 6 --shots 20000 --p 0.001 --pes 0.005 0.01 0.02 0.04 --csv erasure.csv
python plot_figure2.py erasure.csv
```

Sanity check — with `--pes 0` the two decoders must produce identical output and
reproduce the baseline:

```
python erasure_harness.py --rounds 6 --shots 500 --p 0.001 --pes 0
```

This control is what surfaced the duplicate-mechanism problem above. It is worth
running after any change.

Environment used: Stim 1.16.0, ldpc 2.4.1, NumPy 2.5, SciPy, matplotlib 3.11,
Python 3.14, 8-core laptop.

---

## Results

### Baseline

Circuit-level Z memory, 6 rounds, uniform depolarizing, BP-OSD (min-sum,
scaling 0.625, `max_iter` 30, OSD-CS order 10), merged DEM, **50,000 shots per
point**:

| p | fails | LER/round |
| --- | --- | --- |
| 0.0010 | 76 | 2.54e-04 |
| 0.0015 | 378 | 1.26e-03 |
| 0.0020 | 1013 | 3.41e-03 |
| 0.0030 | 4386 | 1.52e-02 |
| 0.0040 | 10977 | 4.05e-02 |

Fitted slope **3.65**. For `d = 6` the usual expectation is around 3 to 3.5, so
this is at the high end. The fit spans `p = 0.001` to `0.004`, part of which is
close to the pseudo-threshold, and the slope has not been fitted with a
confidence interval. It should not be quoted as a measured exponent without
that.

### Heralded erasure

6 rounds, `p = 0.001`, **20,000 shots per point**:

| p_erase | blind fails | aware fails | LER blind | LER aware | BP ok blind | BP ok aware |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0050 | 645 | 36 | 5.45e-03 | 3.00e-04 | 7.0% | 73.2% |
| 0.0075 | 854 | 42 | 7.25e-03 | 3.50e-04 | 3.3% | 73.4% |
| 0.0100 | 1045 | 41 | 8.90e-03 | 3.42e-04 | 2.8% | 73.7% |
| 0.0150 | 1201 | 42 | 1.03e-02 | 3.50e-04 | 2.3% | 73.7% |
| 0.0200 | 957 | 47 | 8.14e-03 | 3.92e-04 | 4.8% | 73.7% |
| 0.0300 | 1181 | 55 | 1.01e-02 | 4.59e-04 | 55.9% | 74.8% |
| 0.0400 | 2206 | 59 | 1.93e-02 | 4.92e-04 | 52.9% | 75.1% |

Three observations.

**1. Heralded erasure is nearly free.** The herald-aware logical error rate
stays within 2x of the erasure-free baseline (2.54e-04 at this `p`) even at an
8x higher erasure rate. The residual failures are set by the unheralded
depolarizing noise, not by erasure.

**2. BP convergence for the aware decoder is flat at ~74%** across the whole
sweep. Informative priors remove the ambiguity regardless of how many erasures
occurred, so the expensive OSD fallback is needed on about a quarter of shots
instead of nearly all of them. Latency follows: in single-core runs,
**62 ms/shot blind vs 14 ms/shot aware**.

**3. The blind decoder is non-monotonic in erasure rate.** Failures fall from
1201 to 957 between `p_e = 0.015` and `0.020` — a 5.3 sigma drop, reproduced in
two independent runs. It coincides with a sharp transition in BP convergence
(2.3% to 4.8% to 55.9% as `p_e` goes 0.015 -> 0.020 -> 0.030). The likely
reading: in this regime the blind decoder's failures are driven by BP
non-convergence rather than by the physical noise level, and as `p_e` grows its
priors on the erasure mechanisms grow with it until BP starts converging. This
is consistent with what the literature already says about BP failure modes on BB
codes. It has not been isolated further here.

### How to read the erasure numbers

The herald-blind decoder is a weak reference point, not a fair baseline. Prior
work argues explicitly that comparing against a decoder which discards known
erasure locations is systematically misleading. The gap above should be read as
*how much information the heralds carry*, not as an improvement over the state
of the art. A fair comparison would use an erasure-aware reference decoder,
which is not implemented here.

### Latency measurement caveat

Per-shot times from `run_parallel.py` run high (roughly 3x) because workers
contend for memory bandwidth. The blind/aware *ratio* is stable across serial
and parallel runs (~4x), but absolute millisecond figures should be taken from
single-core serial runs only.

---

## Limitations

- One code, `[[72,12,6]]`. No distance scaling, no threshold estimate.
- Z-memory only; X-memory not simulated.
- Uniform depolarizing noise, chosen for comparability with published BB-code
  benchmarks rather than for hardware realism.
- Perfect heralding assumed: no false positives or negatives on the erasure flag.
- Latency is wall-clock Python on one core — useful for comparing two decoders
  inside this harness, and roughly four orders of magnitude away from what a
  real-time FPGA or ASIC decoder would need.
- No comparison against an erasure-aware reference decoder.
- Fixed sampler seeds, so repeated runs at the same settings reproduce exactly.
  That is good for reproducibility but means a repeat run is not an independent
  sample.
- The slope (3.65) and the non-monotonicity are reported without further
  investigation.

## Prior work

Checked before writing this up. Every substantive element below already exists
in the literature; this section is what turned this from an attempted paper into
a reproduction project.

- S. Bravyi et al., *High-threshold and low-overhead fault-tolerant quantum
  memory*, Nature 627 (2024) — the BB code construction and benchmark.
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5 (2021).
- J. Roffe et al., *Decoding across the quantum LDPC code landscape*,
  Phys. Rev. Research 2 (2020) — BP-OSD.
- *BiBiEQ: Bivariate Bicycle Codes on Erasure Qubits*, arXiv:2602.07578 —
  compiles BB codes into erasure-aware memory circuits, including imperfect
  heralding. Covers the erasure part of this repository.
- *Fair Decoder Baselines and Rigorous Finite-Size Scaling for Bivariate Bicycle
  Codes on the Quantum Erasure Channel*, arXiv:2603.19062 — argues that
  erasure-ignoring baselines systematically mislead. Read this before
  interpreting the erasure table above.
- M. Gökduman, H. Yao, H. D. Pfister, *Erasure Decoding for Quantum LDPC Codes
  via Belief Propagation with Guided Decimation*, Allerton (2024) — erasure
  information used to reduce BP non-convergence.
- *Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform BP-OSD*,
  arXiv:2507.00254 — measures BP iteration counts and latency on BB codes;
  reduces latency to ~70% of BP-OSD on [[144,12,12]].
- *Best-First Ordered Statistics Decoding of Quantum LDPC Codes*,
  arXiv:2605.25777 — on why conditional OSD invocation is the bottleneck under
  circuit-level noise.
- *Degenerate quantum erasure decoding*, npj Quantum Information (2026).
- *Quasilinear Equivalence Checking for Detector Error Models*,
  arXiv:2606.14677 — formalises DEM scopes and REPEAT-block unrolling; adjacent
  to the duplicate-mechanism note above.

Reducing how often OSD is invoked is an active, competitive research area.
Anyone starting here should read the four decoder papers above first.

## License

MIT — see `LICENSE`.

## Citing

Cite the repository URL and commit hash. There is no paper.
