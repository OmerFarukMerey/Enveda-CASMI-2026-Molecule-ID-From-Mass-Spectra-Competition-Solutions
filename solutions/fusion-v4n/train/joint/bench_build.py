"""Build the fixed joint-model benchmark once: unseen hold-fold molecules, their spectra, their candidate lists, DEV / FINAL halves by scaffold.

Evaluation molecules = the molecules of the fsranker3 table (fold 0 of the identity-disjoint split and the natural-product panel; no
fold-safe model has seen them). Lists per molecule ("views"), exactly the rows the v1 engine produced with the fold-safe FPNet:
  c2N   class-2 list: pool rows of the mass window, the answer among them, its own spectra purged; no PubChem join
  c2J   the same + the 100 joined PubChem-only rows (the production list for a pool answer)
  dropJ answer removed from the pool, joined rows added (the answer, when listed, is a PubChem-only row)
  c1N / c1J  class-1 lists (the molecule's other spectra stay in the library); kept for the ranker refit, not a benchmark view
Only lists that hold the answer are stored (coverage = lists / molecules run, kept in lists.parquet as `n_run`).
The half of a molecule is a hash of its generic Murcko scaffold (acyclic molecules: of their formula), so DEV and FINAL share no scaffold;
`cvfold` (5 scaffold-grouped folds, the folds of the fsranker studies) is for out-of-fold refits inside a half.

in : $CASMI_DATA/rankbank4/data (table), joint/bench/parts/<set>.parquet (candidate strings per table row, scratch/joint0/b0_rows.py),
     the query tables of the studies (spectrum-cache row of every query), rankbank3/replay_{mx,p2}.parquet (fused v33b lists)
out: $CASMI_DATA/joint/bench/{mols,lists,rows,cands,spectra,fused}.parquet, manifest.json.  Refuses to overwrite (the benchmark is fixed).
usage: python train/joint/bench_build.py;  python train/joint/bench_build.py --runs   (coverage table, added afterwards)
"""
import hashlib, json, os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jpaths as P

SALT = 'joint0'
QUERIES = ['ranker_study', 'mixed_regime', 'trainplan', 'panel2/regime', 'rankbank3/hold2']
OURS = ['n2000__s0.npy', 'n2000__s4.npy']          # out-of-fold scores of the ranker fitted on the 5,202 molecules (fsranker3)


def halves(labels):
    """Scaffold label -> 'dev' / 'final'. Scaffolds are taken in the order of a salted hash and each goes to the half that holds fewer
    molecules so far: deterministic, independent of any result, and balanced although a few scaffolds hold hundreds of molecules."""
    cnt = pd.Series(labels).value_counts(); n = dict(dev=0, final=0); out = {}
    for lab in sorted(cnt.index, key=lambda x: hashlib.md5(f'{SALT}:{x}'.encode()).hexdigest()):
        h = 'dev' if n['dev'] <= n['final'] else 'final'; out[lab] = h; n[h] += int(cnt[lab])
    return out


def view_of(kind, joined):
    return {('c2', False): 'c2N', ('c2', True): 'c2J', ('drop', True): 'dropJ', ('c1', False): 'c1N', ('c1', True): 'c1J'}[(kind, bool(joined))]


def runs():
    """runs.parquet: every (set, molecule) the engine was run for, with `listed` (answer among the candidates) -- the coverage denominator.
    Read from the feature caches of the studies (their `mids` hold every molecule with candidates, `empty` those without)."""
    L = pd.read_parquet(P.BENCH + '/lists.parquet'); rows = []
    for st in L.set.drop_duplicates():
        f = f'{P.DATA}/rankbank4/feat/{st.replace("hold2_", "hold2__")}.npz' if st.startswith('hold2_') else f'{P.DATA}/rankbank/feat/c__{st}.npz'
        z = np.load(f, allow_pickle=False); has = set(L.mid[L.set == st])
        rows += [dict(set=st, mid=str(m), candidates=True, listed=str(m) in has) for m in z['mids']] + [dict(set=st, mid=str(m), candidates=False, listed=False) for m in z['empty']]
    d = pd.DataFrame(rows); assert d.listed.sum() == len(L); d.to_parquet(P.BENCH + '/runs.parquet', index=False); return d


def main():
    out = P.BENCH
    if '--runs' in sys.argv:
        print(runs().groupby('set').agg(run=('mid', 'size'), candidates=('candidates', 'sum'), listed=('listed', 'sum'))); return
    if os.path.exists(out + '/manifest.json'):
        sys.exit('benchmark exists: it is fixed, not rebuilt (' + out + ')')
    L = pd.read_parquet(P.TABLE + '/lists.parquet'); M = pd.read_parquet(P.TABLE + '/mols.parquet'); z = np.load(P.TABLE + '/rows.npz')
    names = json.load(open(P.TABLE + '/names.json')); X = np.load(P.TABLE + '/X.npy', mmap_mode='r'); tot = int(L.n.sum()); assert X.shape == (tot, len(names))
    # candidate strings, row-aligned with the table
    parts = pd.concat([pd.read_parquet(f'{out}/parts/{s}.parquet') for s in L.set.drop_duplicates()], ignore_index=True)
    assert (parts.row.values == np.arange(tot)).all()
    pair = parts.key + '\t' + parts.smiles; cand, uniq = pd.factorize(pair)
    first = pd.Series(np.arange(tot)).groupby(cand).first().values
    C = pd.DataFrame(dict(cand=np.arange(len(uniq), dtype=np.int32), key=parts.key.values[first], smiles=parts.smiles.values[first], pid=parts.pid.values[first],
                          formula=parts.formula.values[first], pubchem_only=parts.inj.values[first]))
    lid = np.repeat(np.arange(len(L)), L.n.values)
    kc = pd.factorize(parts.key)[0]                                              # the table's integer key codes are the same partition inside every list
    assert len(np.unique((lid.astype(np.int64) << 32) + kc)) == len(np.unique((lid.astype(np.int64) << 32) + z['key']))
    kind_c2 = np.repeat((L.kind == 'c2').values, L.n.values)
    ours = np.mean([np.load(f'{os.path.dirname(P.TABLE)}/oof/{f}') for f in OURS], axis=0).astype(np.float32)
    R = pd.DataFrame(dict(list=lid.astype(np.int32), cand=cand.astype(np.int32), y=z['y'], inj=z['inj'], fz=z['fz'],
                          fp_cos=np.asarray(X[:, names.index('x__fp_cos')], np.float32),
                          pub=np.where(kind_c2, z['pub_c2'], z['pub_run']).astype(np.float32), ours=ours))
    assert (R.y.values == (C.key.values[cand] == np.repeat(L.mid.map(M.set_index('mid').key).values, L.n.values))).all(), 'answer flag != key match'
    # molecules
    lab = M.scaffold.where(M.scaffold != 'acyclic', 'acyc_' + M.formula.astype(str))
    assert all(len(set(g)) == 1 for g in M.groupby(lab).fold.agg(list))
    st = pd.read_parquet(P.WORK + '/train_structs.parquet', columns=['sid', 'key', 'inchikey14']); sp = pd.read_parquet(P.WORK + '/split.parquet')
    k2 = st.drop_duplicates('key').set_index('key')
    Mo = pd.DataFrame(dict(mid=M.mid, key=M.key, smiles=M.smiles, formula=M.formula, mass=M.mass, scaffold=lab, half=lab.map(halves(lab.values)).values, cvfold=M.fold.astype(np.int8),
                           split_fold=M.key.map(dict(zip(sp.key, sp.fold))).values, sid=M.key.map(k2.sid).values, n_structs=M.key.map(st.groupby('key').size()).values, query_lib=M.qlib_name,
                           np_catalogue=(M.np_cat > 0).values, panel=M.panel.values, enveda=M.qlib_name.isin(['enveda-180', 'enveda-np-examples']).values,
                           nfree=M.nfree.values, batch=M.batch))
    assert Mo.split_fold.isin(['hold', 'np']).all() and Mo.sid.notna().all()
    q = pd.concat([pd.read_parquet(f'{P.DATA}/{d}/queries.parquet', columns=['molecule_id', 'row']) for d in QUERIES]).drop_duplicates()
    q = q[q.molecule_id.isin(set(Mo.mid))].rename(columns=dict(molecule_id='mid')).sort_values(['mid', 'row']).reset_index(drop=True)
    sm = pd.read_parquet(P.WORK + '/spec_meta.parquet', columns=['sid', 'instr'])
    sid2key = np.empty(int(st.sid.max()) + 1, object); sid2key[st.sid.values] = st.key.values      # several structures (tautomers) can share a key
    assert (sid2key[sm.sid.values[q.row.values]] == q.mid.map(Mo.set_index('mid').key).values).all(), 'a query spectrum is not a spectrum of its molecule'
    q['tims'] = sm.instr.values[q.row.values] == 0
    Mo['n_spec'] = Mo.mid.map(q.groupby('mid').size()).fillna(0).astype(int).values; assert (Mo.n_spec > 0).all()
    Mo['tims'] = Mo.mid.map(q.groupby('mid').tims.all()).values
    # lists
    mi = Mo.set_index('mid')
    Lo = pd.DataFrame(dict(list=np.arange(len(L), dtype=np.int32), mid=L.mid, view=[view_of(k, j) for k, j in zip(L.kind, L.joined)], set=L.set, start=L.start, n=L.n,
                           n_run=L.n_lists_in_set, lib_max=L.lib_max, n_isomers=L.fgn, half=L.mid.map(mi.half).values, cvfold=L.mid.map(mi.cvfold).values))
    assert not Lo.duplicated(['mid', 'view', 'set']).any()
    dup = Lo.duplicated(['mid', 'view'], keep=False)                             # a regime molecule has its class-2 list in two sets: keep one as the benchmark list
    Lo['bench'] = ~Lo.duplicated(['mid', 'view'], keep='first') & Lo.view.isin(['c2N', 'c2J', 'dropJ'])
    # fused v33b lists where stored (final submission order after the fusion): regime 380 + panel 250
    fu = []
    for f in ('replay_mx', 'replay_p2'):
        d = pd.read_parquet(f'{P.DATA}/rankbank3/{f}.parquet'); d = d[(d.chain == 'v33b') & d.type.isin(['c2', 'dA'])]
        fu.append(pd.DataFrame(dict(mid=d.mid.values, view=d.type.map(dict(c2='c2J', dA='dropJ')).values, rr=d.rr.values, top=(d.rr.values >= 1.0))))
    fu = pd.concat(fu, ignore_index=True); fu = fu[fu.mid.isin(set(Mo.mid))]
    os.makedirs(out, exist_ok=True)
    for name, d in dict(mols=Mo, lists=Lo, rows=R, cands=C, spectra=q, fused=fu).items():
        d.to_parquet(f'{out}/{name}.parquet', index=False)
    man = dict(built=time.strftime('%Y-%m-%d %H:%M'), salt=SALT, table=P.TABLE, table_rows=tot, features=len(names), molecules=len(Mo), lists=len(Lo), cands=len(C),
               spectra=len(q), duplicate_mid_view_lists=int(dup.sum()), halves=Mo.half.value_counts().to_dict(),
               files={n: hashlib.md5(open(f'{out}/{n}.parquet', 'rb').read()).hexdigest() for n in ('mols', 'lists', 'rows', 'cands', 'spectra', 'fused')})
    json.dump(man, open(out + '/manifest.json', 'w'), indent=1); print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
