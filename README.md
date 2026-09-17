# Circuit-level simulation of the [[72,12,6]] bivariate bicycle code

A small, self-contained, laptop-scale harness for simulating and decoding the
[[72,12,6]] bivariate bicycle (BB) code under circuit-level depolarizing noise,
with optional heralded erasure.

Everything here runs on a single CPU in Python. No quantum hardware, no cluster.

**What this repository is:** an independent reproduction of the circuit-level
memory benchmark for a high-rate qLDPC code, plus one methodological finding
about detector error models that changes the measured numbers by a factor of ~7.

**What this repository is not:** a claim of a new decoding algorithm. Using
heralded-erasure information to inform a decoder is established prior work
(see [Prior work](#prior-work)).

---

## The methodological finding

Stim builds a detector error model (DEM) by bundling errors that are
indistinguishable — same detectors flipped, same effect on the logical
observables — into a single *error mechanism*. That bundling is what makes the
DEM a well-conditioned parity-check matrix for decoding.

When the syndrome-extraction circuit is built by repeating a body:

```python
circuit += (rounds - 1) * body      # emits a REPEAT block
```

the bundling does not happen across the block boundary. The resulting DEM
contains **duplicate columns**: distinct mechanisms with identical signatures.

For the [[72,12,6]] Z-memory at 6 rounds:

| circuit construction | mechanisms | unique signatures | duplicates |
| --- | --- | --- | --- |
| `(rounds-1) * body` (REPEAT) | 3636 | 3096 | **540** |
| fully unrolled | 3096 | 3096 | 0 |

The two circuits are semantically identical — same gates, same noise, same 252
detectors. Only the DEM differs.

Duplicate columns are bad for belief propagation. Two identical columns create a
symmetry BP cannot break, so BP stops converging and OSD has to rescue nearly
every shot. The measured logical error rate comes out far worse than the code
actually is:

| p | LER/round, duplicated DEM | LER/round, merged DEM |
| --- | --- | --- |
| 0.0010 | 9.82e-04 | 2.50e-04 |
| 0.0020 | 6.92e-03 | 2.98e-03 |
| 0.0030 | 2.47e-02 | 1.52e-02 |

The fitted sub-threshold slope moves from **2.92** to **3.70**.

This matters because BB codes are known to lean on OSD precisely *because* of
degeneracy in their Tanner graph. An accidental source of extra degeneracy in
the DEM is easy to introduce and easy to miss, and it silently makes a code look
worse than it is.

`merge_duplicate_mechanisms()` in `bb_harness.py` fixes it. The merge is exact,
not an approximation: two independent mechanisms with the same effect and
probabilities `p1`, `p2` behave as one with probability `p1 + p2 - 2*p1*p2`.

---

## Files

| file | what it does |
| --- | --- |
| `bb_harness.py` | Builds the code, searches for a valid CNOT schedule, builds the Stim circuit, decodes with BP-OSD, sweeps `p` |
| `erasure_harness.py` | Adds heralded erasure; decodes every shot twice (herald-blind vs herald-aware) in a paired comparison |
| `plot_figure1.py` | Log-log LER plot with Wilson-score intervals and a fitted slope |

### `bb_harness.py`

Four stages:

1. **`BBCode`** — constructs the code from `l=6, m=6`, `A = x^3 + y + y^2`,
   `B = y^3 + x + x^2`, giving `HX = [A|B]`, `HZ = [B^T|A^T]`. Self-verifies
   `HX·HZ^T = 0 (mod 2)`, weight-6 checks, degree-6 qubits, `k = 12`.
   `logical_z()` returns 12 independent logical Z operators as
   `ker(HX)` modulo `rowspace(HZ)`.
2. **`find_schedule`** — does *not* assume a CNOT ordering. It searches all
   6!×6! interleaved orderings and keeps those where, for every overlapping
   (X-check, Z-check) pair, an even number of shared data qubits have the
   X-CNOT before the Z-CNOT. That is the condition for the detectors to stay
   deterministic. 9 distinct overlap patterns, 2736 valid schedules; the first
   is used.
3. **`build_z_memory`** — depth-6 interleaved syndrome extraction, uniform
   circuit-level depolarizing noise, final Z checks reconstructed from the data
   measurements.
4. **`run_point`** — samples, converts the DEM to a sparse parity-check matrix,
   merges duplicate mechanisms, decodes with BP-OSD, counts logical failures.

### `erasure_harness.py`

`HERALDED_ERASE(pe)` is applied to the data qubits each round, and each herald
measurement becomes its own `DETECTOR`. In the DEM an erasure then appears as

```
error(pe/2) D_herald
error(pe/2) D_herald D_syndrome...
```

so conditioned on the herald firing, the damaging mechanism has probability 1/2,
and conditioned on it not firing, 0.

Each shot is decoded twice from the *same* sample:

- **herald-blind** — syndrome only; erasure folded into the average rate
- **herald-aware** — herald rows removed from the matrix (they are observed, not
  inferred) and the erasure priors set per shot to 0.5 or ~0

Both decoders share one sampler, so the comparison is paired and no difference
can be attributed to sampling variation.

---

## Install

Python 3.10+.

```
python -m venv venv
# Windows:  .\venv\Scripts\Activate.ps1
# Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
```

## Run

Baseline sweep (Figure 1 data):

```
python bb_harness.py --rounds 6 --shots 50000 --ps 0.001 0.0015 0.002 0.003 0.004 --csv baseline.csv
python plot_figure1.py baseline.csv
```

Erasure comparison:

```
python erasure_harness.py --rounds 6 --shots 3000 --p 0.001 --pes 0.005 0.01 0.02 0.04 --csv erasure.csv
```

Sanity check — with `--pes 0` the two decoders must produce identical output and
reproduce the `bb_harness.py` baseline:

```
python erasure_harness.py --rounds 6 --shots 500 --p 0.001 --pes 0
```

`erasure_harness.py` is much slower than `bb_harness.py`: the decoder priors are
rebuilt on every shot. Start at `--shots 100` and measure before scaling up.

## Reproducing the numbers above

```
python bb_harness.py --rounds 6 --shots 4000 --ps 0.001 0.0015 0.002 0.003 0.004 --csv merged.csv
```

To see the pre-fix behaviour, comment out the `merge_duplicate_mechanisms` call
at the end of `dem_to_matrices`.

Environment used: Stim 1.16.0, ldpc 2.4.1, NumPy 2.5, SciPy, matplotlib 3.11,
Python 3.14, single CPU core.

---

## Results

### Baseline

Circuit-level Z memory, 6 rounds, uniform depolarizing, BP-OSD
(min-sum, scaling 0.625, `max_iter` 30, OSD-CS order 10), merged DEM,
4000 shots per point:

| p | fails | LER/round |
| --- | --- | --- |
| 0.0010 | 6 | 2.50e-04 |
| 0.0015 | 23 | 9.61e-04 |
| 0.0020 | 71 | 2.98e-03 |
| 0.0030 | 352 | 1.52e-02 |
| 0.0040 | 830 | 3.80e-02 |

Fitted slope 3.70. A distance-6 code is expected near 3–3.5, so this slightly
overshoots. The most likely cause is the leftmost point: 6 failures in 4000
shots is far too few to pin down, and that point dominates the fit. Higher
statistics at low `p` are needed before quoting a slope.

### Heralded erasure

6 rounds, `p = 0.001`, 3000 shots per point:

| p_erase | blind fails | aware fails | ms/shot blind | ms/shot aware |
| --- | --- | --- | --- | --- |
| 0.005 | 96 | 6 | 123 | 34 |
| 0.010 | 179 | 7 | 125 | 31 |
| 0.020 | 153 | 12 | 123 | 32 |
| 0.040 | 337 | 9 | 56 | 31 |

Two observations. Herald-aware decoding is far more accurate, and it is also
*faster* — informative priors let BP converge, so the expensive OSD fallback is
needed much less often.

**How to read these numbers.** The herald-blind decoder is a weak reference
point, not a fair baseline: prior work argues explicitly that comparing against
a decoder that discards known erasure locations is systematically misleading.
The accuracy gap above should therefore be read as *how much information the
heralds carry*, not as an improvement over the state of the art. A fair
comparison would put this against an erasure-aware reference decoder, which is
not implemented here.

---

## Limitations

- One code, `[[72,12,6]]`. No distance scaling, no threshold estimate.
- Z-memory only; X-memory not simulated.
- Uniform depolarizing noise. Real devices have biased, correlated and
  non-uniform errors; the uniform model is chosen for comparability with
  published BB-code benchmarks, not for realism.
- Perfect heralding assumed: no false positives or negatives on the erasure flag.
- Latency is wall-clock Python on one CPU core. It is useful for comparing two
  decoders against each other in this harness and says nothing about a
  real-time FPGA or ASIC decoder.
- Statistics are thin at low `p` and for the herald-aware decoder (single-digit
  failure counts). Error bars are wide; the slope is not yet reliable.
- No comparison against an erasure-aware reference decoder.

## Prior work

Using heralded erasure to inform a decoder, including on bivariate bicycle
codes, is established. Anyone building on this repository should read:

- S. Bravyi et al., *High-threshold and low-overhead fault-tolerant quantum
  memory*, Nature 627 (2024) — the BB code construction and benchmark.
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5 (2021).
- J. Roffe et al., *Decoding across the quantum low-density parity-check code
  landscape*, Phys. Rev. Research 2 (2020) — BP-OSD.
- *BiBiEQ: Bivariate Bicycle Codes on Erasure Qubits*, arXiv:2602.07578 —
  compiles BB codes into erasure-aware memory circuits; directly overlaps the
  erasure part of this repository.
- *Fair Decoder Baselines and Rigorous Finite-Size Scaling for Bivariate
  Bicycle Codes on the Quantum Erasure Channel*, arXiv:2603.19062 — argues that
  erasure-ignoring baselines systematically mislead. Read this before
  interpreting the erasure table above.
- *Degenerate quantum erasure decoding*, npj Quantum Information (2026) —
  linear-time BP decoders exploiting degeneracy, near capacity.
- *Degeneracy Cutting*, arXiv:2510.08695 — post-processing for BP on qLDPC codes
  and the degeneracy problem in BB codes.

## License

MIT — see `LICENSE`.

## Citing

If this is useful, cite the repository URL and commit hash. There is no paper.
