"""
Stage-1 harness: circuit-level baseline for the [[72,12,6]] bivariate bicycle code.

What it does
------------
1. Builds the [[72,12,6]] BB code (Bravyi et al., Nature 2024) and VERIFIES n, k,
   stabilizer commutation and check weights.
2. Builds a Stim circuit-level Z-memory experiment with a validated depth-6
   interleaved CNOT schedule (the schedule is searched for, not assumed).
3. Decodes with BP-OSD (baseline) and reports logical error rate per round.

This is the foundation. Once the baseline curve is verified, the erasure-aware
decoder replaces ONLY the decode step in run_point().

Install:  pip install stim sinter pymatching ldpc scipy numpy
Run:      python bb_harness.py --rounds 6 --shots 2000 --ps 0.0005 0.001 0.002

EXPECTED OUTPUT (shape, not exact numbers -- these are stochastic):

    === code verification ===
    [[72,12,6]] bivariate bicycle   X-checks=36  Z-checks=36  weight=6  degree=6
    commutation OK, k=12 OK

    === schedule ===
    X order (0, 1, 2, 3, 4, 5)   Z order (2, 1, 0, 5, 4, 3)   (validated against 9 overlap patterns)

    === circuit-level Z memory, 6 rounds ===
           p   shots  fails    LER/round          +-
      0.0005     200      0    0.000e+00    1.2e-08
      0.0010     200      2    1.674e-03    1.2e-03
      0.0020     200      6    5.064e-03    2.0e-03

The three things that MUST be true:
  1. code verification prints [[72,12,6]] and passes both asserts
  2. Stim accepts the circuit (a non-deterministic detector raises an error here)
  3. LER/round falls monotonically and steeply as p falls  <- this is the signal
     that error suppression is real and the pipeline is correct

Note on comparing to Bravyi et al. (Nature 2024): this file uses a uniform
depolarizing model with idle noise on every sub-layer, which is harsher than
theirs, and counts a failure if ANY of the 12 logical qubits fails. Expect the
same shape and threshold region, not identical numbers. Matching their model
exactly is the last step of baseline reproduction.
"""
import itertools
import numpy as np
import stim
from scipy.sparse import csc_matrix
from ldpc import BpOsdDecoder

# ---------------------------------------------------------------- code
def _shift(n):
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S

def _mpow(M, k):
    r = np.eye(M.shape[0], dtype=np.uint8)
    for _ in range(k):
        r = (r @ M) % 2
    return r

def _rank2(M):
    M = M.copy() % 2
    r = 0
    for c in range(M.shape[1]):
        piv = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r

def _nullspace2(M):
    """Basis of {v : M v = 0 mod 2}, returned as rows."""
    M = M.copy() % 2
    rows, cols = M.shape
    piv_cols, r = [], 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv_cols.append(c)
        r += 1
    free = [c for c in range(cols) if c not in piv_cols]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, pc in enumerate(piv_cols):
            v[pc] = M[i, f]
        basis.append(v)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, cols), np.uint8)

def _row_reduce_against(V, W):
    """Return rows of V that are independent modulo rowspace(W)."""
    out = []
    # gaussian-reduce the existing basis first
    red = []
    for b in W:
        v = b.copy()
        for r in red:
            p = np.argmax(r)
            if r[p] and v[p]:
                v ^= r
        if v.any():
            red.append(v)
    basis = red
    for v in V:
        u = v.copy()
        for r in basis:
            p = np.argmax(r)
            if r[p] and u[p]:
                u ^= r
        if u.any():
            basis.append(u)
            out.append(v)
    return np.array(out, dtype=np.uint8)

class BBCode:
    def __init__(self, l=6, m=6, a_exp=(("x", 3), ("y", 1), ("y", 2)),
                 b_exp=(("y", 3), ("x", 1), ("x", 2))):
        self.l, self.m = l, m
        self.N = l * m
        self.n = 2 * l * m
        x = np.kron(_shift(l), np.eye(m, dtype=np.uint8))
        y = np.kron(np.eye(l, dtype=np.uint8), _shift(m))
        base = {"x": x, "y": y}
        self.A_terms = [_mpow(base[v], e) for v, e in a_exp]
        self.B_terms = [_mpow(base[v], e) for v, e in b_exp]
        A = np.zeros_like(x); B = np.zeros_like(x)
        for T in self.A_terms: A = (A + T) % 2
        for T in self.B_terms: B = (B + T) % 2
        self.A, self.B = A, B
        self.HX = np.hstack([A, B]) % 2
        self.HZ = np.hstack([B.T, A.T]) % 2
        self.k = self.n - _rank2(self.HX) - _rank2(self.HZ)

    def verify(self):
        assert np.all((self.HX @ self.HZ.T) % 2 == 0), "stabilizers do not commute"
        assert set(self.HX.sum(1)) == {6} and set(self.HZ.sum(1)) == {6}, "bad check weight"
        return dict(n=self.n, k=self.k,
                    x_checks=self.HX.shape[0], z_checks=self.HZ.shape[0],
                    check_weight=6, qubit_degree=int((self.HX.sum(0) + self.HZ.sum(0))[0]))

    def logical_z(self):
        """12 independent logical Z operators: ker(HX) modulo rowspace(HZ)."""
        ker = _nullspace2(self.HX)
        return _row_reduce_against(ker, self.HZ)[: self.k]

    # qubit touched by X-check i / Z-check j for each of the 6 CNOT terms
    def x_qubit(self, i, t):
        M = self.A_terms[t] if t < 3 else self.B_terms[t - 3]
        q = int(np.nonzero(M[i])[0][0])
        return q if t < 3 else self.N + q

    def z_qubit(self, j, t):
        M = self.B_terms[t].T if t < 3 else self.A_terms[t - 3].T
        q = int(np.nonzero(M[j])[0][0])
        return q if t < 3 else self.N + q

# ------------------------------------------------------- CNOT schedule
def find_schedule(code):
    """Search for an interleaved depth-6 schedule whose detectors stay deterministic.

    Condition: for every overlapping (X-check, Z-check) pair, the number of shared
    data qubits where the X-CNOT precedes the Z-CNOT must be even.
    """
    patterns = set()
    for i in range(code.N):
        xs = {code.x_qubit(i, t): t for t in range(6)}
        for j in range(code.N):
            pairs = [(xs[code.z_qubit(j, t)], t) for t in range(6)
                     if code.z_qubit(j, t) in xs]
            if pairs:
                patterns.add(tuple(sorted(pairs)))
    for pX in itertools.permutations(range(6)):
        for pZ in itertools.permutations(range(6)):
            if all(sum(1 for tx, tz in pat if pX[tx] < pZ[tz]) % 2 == 0
                   for pat in patterns):
                return pX, pZ, len(patterns)
    raise RuntimeError("no valid schedule found")

# ------------------------------------------------------------ circuit
def build_z_memory(code, rounds, p, schedule):
    """Circuit-level Z-memory experiment (detects X errors). Uniform depolarizing model."""
    pX, pZ = schedule
    n, N = code.n, code.N
    xa = [n + i for i in range(N)]          # X-check ancillas
    za = [n + N + j for j in range(N)]      # Z-check ancillas
    data = list(range(n))
    c = stim.Circuit()
    c.append("R", data + xa + za)
    c.append("X_ERROR", data + xa + za, p)
    c.append("TICK")

    def one_round(circ):
        circ.append("H", xa)
        circ.append("DEPOLARIZE1", xa, p)
        for s in range(6):
            targets, busy = [], set()
            for i in range(N):
                t = pX.index(s)
                targets += [xa[i], code.x_qubit(i, t)]
                busy |= {xa[i], code.x_qubit(i, t)}
            for j in range(N):
                t = pZ.index(s)
                targets += [code.z_qubit(j, t), za[j]]
                busy |= {code.z_qubit(j, t), za[j]}
            circ.append("CX", targets)
            circ.append("DEPOLARIZE2", targets, p)
            idle = [q for q in data + xa + za if q not in busy]
            if idle:
                circ.append("DEPOLARIZE1", idle, p)
            circ.append("TICK")
        circ.append("H", xa)
        circ.append("DEPOLARIZE1", xa, p)
        circ.append("X_ERROR", xa + za, p)
        circ.append("M", xa + za)
        circ.append("R", xa + za)
        circ.append("X_ERROR", xa + za, p)
        circ.append("DEPOLARIZE1", data, p)
        circ.append("TICK")

    one_round(c)
    # round 0: Z-check outcomes are deterministic (data initialised in |0>)
    for j in range(N):
        c.append("DETECTOR", [stim.target_rec(-N + j)])

    body = stim.Circuit()
    one_round(body)
    for j in range(N):
        body.append("DETECTOR", [stim.target_rec(-N + j),
                                 stim.target_rec(-3 * N + j)])
    c += (rounds - 1) * body

    c.append("M", data)
    # final Z checks reconstructed from the data measurements
    for j in range(N):
        recs = [stim.target_rec(-n + q) for q in np.nonzero(code.HZ[j])[0]]
        recs.append(stim.target_rec(-n - N + j))
        c.append("DETECTOR", recs)
    for idx, lz in enumerate(code.logical_z()):
        c.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(-n + int(q)) for q in np.nonzero(lz)[0]], idx)
    return c

# ------------------------------------------------------------ decoding
def dem_to_matrices(dem):
    """DEM -> (check matrix H, observable matrix L, error priors)."""
    H_rows, H_cols, L_rows, L_cols, priors = [], [], [], [], []
    e = 0
    def handle(inst):
        nonlocal e
        if inst.type == "error":
            dets, obs = [], []
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    obs.append(t.val)
            for d in dets:
                H_rows.append(d); H_cols.append(e)
            for o in obs:
                L_rows.append(o); L_cols.append(e)
            priors.append(inst.args_copy()[0])
            e += 1
        elif inst.type == "repeat":
            for _ in range(inst.repeat_count):
                for sub in inst.body_copy():
                    handle(sub)
    for inst in dem.flattened():
        handle(inst)
    H = csc_matrix((np.ones(len(H_rows), np.uint8), (H_rows, H_cols)),
                   shape=(dem.num_detectors, e))
    L = csc_matrix((np.ones(len(L_rows), np.uint8), (L_rows, L_cols)),
                   shape=(dem.num_observables, e))
    return merge_duplicate_mechanisms(H, L, np.array(priors))

def merge_duplicate_mechanisms(H, L, priors):
    """Collapse error mechanisms with identical (detector, observable) signatures.

    Stim cannot merge identical mechanisms across a REPEAT block, so a DEM taken
    from a circuit built with `(rounds-1) * body` contains duplicate columns --
    for the [[72,12,6]] Z-memory at 6 rounds, 540 of 3636 columns are duplicates.
    Duplicate columns create a symmetry that belief propagation cannot break: BP
    stops converging, OSD has to rescue every shot, and the measured logical error
    rate comes out several times worse than the code actually is. Merging them is
    exact, not an approximation: two independent mechanisms with the same effect
    and probabilities p1, p2 act as one with probability p1 + p2 - 2*p1*p2.
    """
    Hc, Lc = H.tocsc(), L.tocsc()
    groups = {}
    for j in range(Hc.shape[1]):
        sig = (tuple(sorted(int(r) for r in Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]])),
               tuple(sorted(int(r) for r in Lc.indices[Lc.indptr[j]:Lc.indptr[j + 1]])))
        groups.setdefault(sig, []).append(j)

    H_rows, H_cols, L_rows, L_cols, merged = [], [], [], [], []
    for new_j, (sig, cols) in enumerate(groups.items()):
        dets, obs = sig
        for d in dets:
            H_rows.append(d); H_cols.append(new_j)
        for o in obs:
            L_rows.append(o); L_cols.append(new_j)
        q = 0.0
        for j in cols:                       # XOR-combine independent duplicates
            q = q + priors[j] - 2 * q * priors[j]
        merged.append(q)

    E = len(groups)
    Hm = csc_matrix((np.ones(len(H_rows), np.uint8), (H_rows, H_cols)),
                    shape=(H.shape[0], E))
    Lm = csc_matrix((np.ones(len(L_rows), np.uint8), (L_rows, L_cols)),
                    shape=(L.shape[0], E))
    return Hm, Lm, np.array(merged)

def run_point(code, p, rounds, shots, schedule, osd_order=10, seed=0):
    circ = build_z_memory(code, rounds, p, schedule)
    dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    H, L, priors = dem_to_matrices(dem)
    dec = BpOsdDecoder(H, error_channel=list(priors), max_iter=30,
                       bp_method="ms", ms_scaling_factor=0.625,
                       schedule="parallel", osd_method="osd_cs", osd_order=osd_order)
    sampler = circ.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    fails = 0
    for s in range(shots):
        corr = dec.decode(dets[s].astype(np.uint8))
        pred = (L @ corr) % 2
        if np.any(pred != obs[s]):
            fails += 1
    return fails, shots, circ.num_detectors, dem.num_errors

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    import argparse, math
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.002, 0.003])
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    code = BBCode()
    info = code.verify()
    print("=== code verification ===")
    print(f"[[{info['n']},{info['k']},6]] bivariate bicycle   "
          f"X-checks={info['x_checks']}  Z-checks={info['z_checks']}  "
          f"weight={info['check_weight']}  degree={info['qubit_degree']}")
    assert (info["n"], info["k"]) == (72, 12), "code parameters wrong"
    print("commutation OK, k=12 OK")

    pX, pZ, npat = find_schedule(code)
    print(f"\n=== schedule ===\nX order {pX}   Z order {pZ}   "
          f"(validated against {npat} overlap patterns)")

    print(f"\n=== circuit-level Z memory, {args.rounds} rounds ===")
    print(f"{'p':>8} {'shots':>7} {'fails':>6} {'LER/round':>12} {'±':>10}")
    rows = []
    for p in args.ps:
        f, s, nd, ne = run_point(code, p, args.rounds, args.shots, (pX, pZ))
        ler = f / s
        per_round = 1 - (1 - ler) ** (1 / args.rounds)
        err = math.sqrt(max(ler * (1 - ler), 1e-12) / s) / args.rounds
        print(f"{p:>8.4f} {s:>7} {f:>6} {per_round:>12.3e} {err:>10.1e}")
        rows.append((p, args.rounds, s, f, per_round, err, nd, ne))

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("p,rounds,shots,fails,ler_per_round,stderr,num_detectors,num_dem_errors\n")
            for r in rows:
                fh.write(",".join(str(v) for v in r) + "\n")
        print(f"\nwrote {args.csv}")
