"""Per-molecule analog candidates with gate features, on held-out training structures.
usage: eval2.py LIB_DIR N_QUERIES SEED OUT_PARQUET [SOURCE]
For each sampled structure its own spectra are the queries and every other structure is the library. Each of the top
analogs (distinct structure x modification, best modified cosine over the molecule's spectra) that has at least one
valid edit becomes a row with features and labels: truth among its edits, truth = its top site pick."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ae_core as A

TOPK, MIN_SCORE = 5, 0.6


def search_all(qm, qi, nq, qprec, g, L, exclude, prec_tol=0.01, tol=0.01):
    """Every (score, row, kind, direction, delta) with modified cosine >= MIN_SCORE."""
    rows, precs = L.by[g]
    out = []
    for kind, delta, direction in A.TARGETS:
        lo, hi = np.searchsorted(precs, [qprec - delta - prec_tol, qprec - delta + prec_tol])
        for r in rows[lo:hi]:
            if L.n[r] < 3 or L.sidx[r] == exclude:
                continue
            s = A.modcos(qm, qi, nq, L.mz[r], L.it[r], L.n[r], delta, tol)
            if s >= MIN_SCORE:
                out.append((s, int(r), kind, direction, float(delta)))
    return out


def pair_stats(qm, qi, am, ai, delta):
    pairs = A.align(qm, qi, am, ai, delta)
    tot = float(np.sqrt((qi ** 2).sum() * (ai ** 2).sum()))
    sh = [w for _, s, w in pairs if s]; un = [w for _, s, w in pairs if not s]
    return dict(n_match=len(pairs), n_shift=len(sh), w_shift=sum(sh) / tot, w_unshift=sum(un) / tot)


def molecule_rows(s, L, src, code, rows_of):
    qrows = [r for r in rows_of[s] if src[r] == code and L.n[r] >= 3]
    hits = {}
    for q in qrows:
        for sc, r, kind, direction, delta in search_all(L.mz[q][:L.n[q]], L.it[q][:L.n[q]], L.n[q], L.prec[q], int(L.grp[q]), L, s):
            k = (int(L.sidx[r]), kind, direction)
            h = hits.setdefault(k, dict(best=0.0, spectra=set(), scores=[]))
            h['scores'].append(sc); h['spectra'].add(q)
            if sc > h['best']:
                h.update(best=sc, row=r, q=q, delta=delta)
    out = []
    ranked = sorted(hits.items(), key=lambda kv: -kv[1]['best'])
    for rank, ((a, kind, direction), h) in enumerate(ranked[:TOPK]):
        # a product that is another library structure has its own spectra; on the test it would be a library match,
        # so the overlay drops it (here the held-out truth stays, as a test answer is not in the library)
        prods = {k: v for k, v in A.edits(L.smiles[a], kind, direction).items() if k not in LIBKEYS or k == L.keys[s]}
        if not prods:
            continue
        q, ar = h['q'], h['row']
        qm, qi, am, ai = L.mz[q][:L.n[q]], L.it[q][:L.n[q]], L.mz[ar][:L.n[ar]], L.it[ar][:L.n[ar]]
        try:
            rp = A.rank_products(L.smiles[a], prods, am, ai, qm, qi, h['delta'], int(L.grp[q]))
        except Exception:  # noqa: BLE001
            continue
        truth = L.keys[s]
        out.append(dict(sidx=int(s), cand_rank=rank, analog=L.keys[a], kind=kind, direction=direction,
                        abs_delta=abs(h['delta']), score=h['best'], n_qspec=len(qrows), n_spec_hit=len(h['spectra']),
                        mean_score=float(np.mean(h['scores'])), second=ranked[rank + 1][1]['best'] if rank + 1 < len(ranked) else 0.0,
                        top_score=ranked[0][1]['best'], n_prod=len(prods), site1=rp[0][0],
                        site_margin=rp[0][0] - (rp[1][0] if len(rp) > 1 else 0.0), prec=float(L.prec[q]), grp=int(L.grp[q]),
                        nq=int(L.n[q]), na=int(L.n[ar]), **pair_stats(qm, qi, am, ai, h['delta']),
                        truth_in=truth in prods, pick_ok=rp[0][1] == truth, pick=rp[0][1], pick_smiles=rp[0][2]))
    return out


if __name__ == '__main__':
    lib, n_q, seed, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    source = sys.argv[5] if len(sys.argv) > 5 else 'enveda-180'
    L = A.Library(lib / 'lib_peaks.npz', pd.read_parquet(lib / 'lib_struct.parquet'))
    LIBKEYS = set(L.keys)
    src = np.load(lib / 'lib_peaks.npz')['src']
    code = json.loads((lib / 'lib_sources.json').read_text())[source]
    pool = np.unique(L.sidx[src == code])
    picks = np.random.default_rng(seed).choice(pool, min(n_q, len(pool)), replace=False)
    rows_of = pd.Series(np.arange(len(L.sidx))).groupby(L.sidx).apply(np.array)
    t0 = time.time(); recs = []
    for i, s in enumerate(picks):
        recs += molecule_rows(int(s), L, src, code, rows_of) or [dict(sidx=int(s), cand_rank=-1)]
        if (i + 1) % 250 == 0:
            print(f'{i + 1} molecules, {len(recs)} rows, {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(recs).to_parquet(out, index=False)
    print(f'done {len(picks)} molecules, {len(recs)} rows in {time.time() - t0:.0f}s')
