"""
Figure 2 generator: erasure-aware decoding of the [[72,12,6]] bivariate bicycle code.

THE EXPERIMENT
--------------
Neutral-atom hardware sometimes loses a qubit outright (the atom leaves the trap).
The machine KNOWS which qubit was lost -- it just does not know what value it held.
That "herald" is free side information. Standard decoders throw it away.

This script runs ONE set of noisy shots and decodes each shot TWICE:

  (a) erasure-ignoring  -- the decoder gets the syndrome only (this is the baseline
                           practice: erasure is folded into the average error rate)
  (b) erasure-aware     -- the decoder additionally gets the herald bits, and raises
                           the prior probability on exactly the erased locations

Because both decoders see IDENTICAL shots, the comparison is paired: any difference
in logical error rate is caused by the side information alone, not by sampling luck.

We also time each decode, because a decoder that is accurate but too slow is useless
in a real fault-tolerant machine.

HOW THE HERALD IS WIRED IN
--------------------------
Stim's HERALDED_ERASE(pe) emits a measurement bit that is 1 exactly when the erasure
fired. We turn each of those bits into its own DETECTOR. In the detector error model
an erasure then shows up as two mechanisms:

    error(pe/2) D_herald                 <- erased, but no observable flip
    error(pe/2) D_herald D_syndrome...   <- erased, and it did flip things

So conditioned on the herald being 1, the damaging mechanism has probability 1/2,
and conditioned on the herald being 0 it has probability 0. That is the entire
contribution: per shot, set those priors to 0.5 or to ~0 instead of to pe/2.

Usage:
    python erasure_harness.py --rounds 6 --shots 2000 --p 0.001 --pes 0.001 0.002 0.004 --csv erasure.csv

Start small (--shots 200) -- this is much slower than bb_harness.py because the
decoder priors are rebuilt on every single shot.
"""
import argparse
import math
import time

import numpy as np
import stim
from scipy.sparse import csc_matrix
from ldpc import BpOsdDecoder

from bb_harness import BBCode, find_schedule, dem_to_matrices


# ------------------------------------------------------------------ circuit
def build_z_memory_erasure(code, rounds, p, pe, schedule):
    """Z-memory circuit with heralded erasure on the data qubits each round.

    Returns (circuit, herald_detector_indices) where herald_detector_indices[r][q]
    is the detector index carrying the herald for data qubit q in round r.
    """
    pX, pZ = schedule
    n, N = code.n, code.N
    xa = [n + i for i in range(N)]
    za = [n + N + j for j in range(N)]
    data = list(range(n))

    c = stim.Circuit()
    c.append("R", data + xa + za)
    c.append("X_ERROR", data + xa + za, p)
    c.append("TICK")

    ndet = 0                 # running detector index, in order of appearance
    herald_dets = []         # herald_dets[r][q] -> detector index

    def syndrome_extraction():
        c.append("H", xa)
        c.append("DEPOLARIZE1", xa, p)
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
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p)
            idle = [q for q in data + xa + za if q not in busy]
            if idle:
                c.append("DEPOLARIZE1", idle, p)
            c.append("TICK")
        c.append("H", xa)
        c.append("DEPOLARIZE1", xa, p)
        c.append("X_ERROR", xa + za, p)
        c.append("M", xa + za)
        c.append("R", xa + za)
        c.append("X_ERROR", xa + za, p)
        c.append("DEPOLARIZE1", data, p)
        c.append("TICK")

    for r in range(rounds):
        # --- erasure layer: n heralded erasures, n herald measurement records ---
        if pe > 0:
            c.append("HERALDED_ERASE", data, pe)
            this_round = []
            for q in range(n):
                c.append("DETECTOR", [stim.target_rec(-n + q)])
                this_round.append(ndet)
                ndet += 1
            herald_dets.append(this_round)
        else:
            herald_dets.append([])

        syndrome_extraction()

        if r == 0:
            # data started in |0>, so round-0 Z outcomes are deterministic
            for j in range(N):
                c.append("DETECTOR", [stim.target_rec(-N + j)])
                ndet += 1
        else:
            # compare with the same Z check one round earlier.
            # layout going back from here: [2N this round][n heralds][2N last round]
            for j in range(N):
                c.append("DETECTOR", [stim.target_rec(-N + j),
                                      stim.target_rec(-3 * N - n + j)])
                ndet += 1

    c.append("M", data)
    for j in range(N):
        recs = [stim.target_rec(-n + int(q)) for q in np.nonzero(code.HZ[j])[0]]
        recs.append(stim.target_rec(-n - N + j))
        c.append("DETECTOR", recs)
        ndet += 1
    for idx, lz in enumerate(code.logical_z()):
        c.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(-n + int(q)) for q in np.nonzero(lz)[0]], idx)

    assert ndet == c.num_detectors, (ndet, c.num_detectors)
    return c, herald_dets


# ------------------------------------------------------------------ decoding
def split_by_herald(H, L, priors, herald_rows, num_detectors):
    """Separate herald detectors from syndrome detectors.

    Returns:
        H_syn        H with the herald rows deleted (what the decoder explains)
        keep         column mask of mechanisms worth keeping
        mech_herald  for each kept mechanism, its herald detector index or -1
        syn_rows     indices of the real syndrome detectors
    """
    herald_set = set(herald_rows)
    syn_rows = np.array([d for d in range(num_detectors) if d not in herald_set])

    Hc = H.tocsc()
    E = Hc.shape[1]
    mech_herald = np.full(E, -1, dtype=np.int64)
    for j in range(E):
        rows = Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]
        hits = [int(r) for r in rows if int(r) in herald_set]
        if len(hits) > 1:
            raise RuntimeError("mechanism touches two heralds -- unexpected")
        if hits:
            mech_herald[j] = hits[0]

    H_syn = Hc[syn_rows, :].tocsc()
    Lc = L.tocsc()

    # a mechanism that flips nothing but its own herald is invisible: drop it
    touches_syn = np.diff(H_syn.indptr) > 0
    touches_obs = np.diff(Lc.indptr) > 0
    keep = touches_syn | touches_obs

    return (H_syn[:, keep].tocsc(), Lc[:, keep].tocsc(),
            priors[keep], mech_herald[keep], syn_rows)


def make_decoder(H, priors, osd_order):
    return BpOsdDecoder(H, error_channel=list(priors), max_iter=30,
                        bp_method="ms", ms_scaling_factor=0.625,
                        schedule="parallel", osd_method="osd_cs",
                        osd_order=osd_order)


def run_point(code, p, pe, rounds, shots, schedule, osd_order=10, seed=0):
    circ, herald_dets = build_z_memory_erasure(code, rounds, p, pe, schedule)
    dem = circ.detector_error_model(decompose_errors=False,
                                    approximate_disjoint_errors=True)
    H, L, priors = dem_to_matrices(dem)

    herald_rows = [d for rd in herald_dets for d in rd]
    H_syn, L_k, priors_k, mech_herald, syn_rows = split_by_herald(
        H, L, priors, herald_rows, circ.num_detectors)

    erased_mask = mech_herald >= 0
    herald_of = mech_herald[erased_mask]

    dec_blind = make_decoder(H_syn, priors_k, osd_order)
    dec_aware = make_decoder(H_syn, priors_k, osd_order)

    sampler = circ.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)

    shot_priors = priors_k.copy()
    fails_blind = fails_aware = 0
    t_blind = t_aware = 0.0

    for s in range(shots):
        full = dets[s]
        syn = full[syn_rows].astype(np.uint8)

        # (a) erasure-ignoring
        t0 = time.perf_counter()
        corr = dec_blind.decode(syn)
        t_blind += time.perf_counter() - t0
        if np.any((L_k @ corr) % 2 != obs[s]):
            fails_blind += 1

        # (b) erasure-aware: 0.5 where the herald fired, ~0 where it did not
        t0 = time.perf_counter()
        if erased_mask.any():
            fired = full[herald_of]
            shot_priors[erased_mask] = np.where(fired, 0.5, 1e-9)
            dec_aware.update_channel_probs(list(shot_priors))
        corr = dec_aware.decode(syn)
        t_aware += time.perf_counter() - t0
        if np.any((L_k @ corr) % 2 != obs[s]):
            fails_aware += 1

    return dict(fails_blind=fails_blind, fails_aware=fails_aware, shots=shots,
                ms_blind=1e3 * t_blind / shots, ms_aware=1e3 * t_aware / shots,
                num_erasure_mechs=int(erased_mask.sum()),
                num_detectors=circ.num_detectors, num_errors=dem.num_errors)


def per_round(fails, shots, rounds):
    ler = fails / shots
    return 1 - (1 - ler) ** (1 / rounds) if ler > 0 else 0.0


def stderr_per_round(fails, shots, rounds):
    ler = fails / shots
    return math.sqrt(max(ler * (1 - ler), 1e-12) / shots) / rounds


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--p", type=float, default=0.001,
                    help="ordinary circuit-level depolarizing rate")
    ap.add_argument("--pes", type=float, nargs="+", default=[0.001, 0.002, 0.004],
                    help="heralded erasure rates to sweep")
    ap.add_argument("--osd-order", type=int, default=10)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    code = BBCode()
    info = code.verify()
    print("=== code verification ===")
    print(f"[[{info['n']},{info['k']},6]] bivariate bicycle   "
          f"X-checks={info['x_checks']}  Z-checks={info['z_checks']}  "
          f"weight={info['check_weight']}  degree={info['qubit_degree']}")
    print("commutation OK, k=12 OK\n")

    schedule = find_schedule(code)
    pX, pZ, npat = schedule[0], schedule[1], schedule[2]
    print("=== schedule ===")
    print(f"X order {pX}   Z order {pZ}   (validated against {npat} overlap patterns)\n")

    print(f"=== erasure-aware vs erasure-ignoring decoding, "
          f"{args.rounds} rounds, p={args.p} ===")
    print(f"{'p_erase':>8} {'shots':>7} {'blind':>7} {'aware':>7} "
          f"{'LER blind':>11} {'LER aware':>11} {'gain':>6} "
          f"{'ms blind':>9} {'ms aware':>9}")

    rows = []
    for pe in args.pes:
        r = run_point(code, args.p, pe, args.rounds, args.shots, (pX, pZ),
                      osd_order=args.osd_order)
        lb = per_round(r["fails_blind"], r["shots"], args.rounds)
        la = per_round(r["fails_aware"], r["shots"], args.rounds)
        gain = (lb / la) if la > 0 else float("inf")
        print(f"{pe:>8.4f} {r['shots']:>7} {r['fails_blind']:>7} {r['fails_aware']:>7} "
              f"{lb:>11.3e} {la:>11.3e} {gain:>6.2f} "
              f"{r['ms_blind']:>9.2f} {r['ms_aware']:>9.2f}")
        rows.append((args.p, pe, args.rounds, r["shots"],
                     r["fails_blind"], r["fails_aware"], lb, la,
                     stderr_per_round(r["fails_blind"], r["shots"], args.rounds),
                     stderr_per_round(r["fails_aware"], r["shots"], args.rounds),
                     r["ms_blind"], r["ms_aware"]))

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("p,p_erase,rounds,shots,fails_blind,fails_aware,"
                     "ler_blind,ler_aware,stderr_blind,stderr_aware,"
                     "ms_blind,ms_aware\n")
            for row in rows:
                fh.write(",".join(str(v) for v in row) + "\n")
        print(f"\nwrote {args.csv}")
