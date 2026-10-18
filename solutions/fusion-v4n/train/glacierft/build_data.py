"""GLACIER fine-tune data: scaffold-disjoint held-out / validation / training split of train.parquet and the runner-format isomer items.

Sets (molecule = metric key, the tautomer-canonical InChIKey-14 `key` of the spectrum cache's train_structs):
  R   every stack1-like replay molecule (mixed regime 380 + Enveda panel 226; scratch/glacierft/r_meta.py). Never trained on.
  H1  R's timsTOF molecules in stack1's GLACIER branch with the truth among the scored candidates, NOT in MassSpecGym; group =
      the candidates stack1 scored with the truth's formula (tims study groups).
  H2  Enveda-180 timsTOF molecules with an [M+H]+ spectrum, NOT in MassSpecGym and not in R, first N_H by a salted hash; group =
      truth + the N_DEC same-formula pool structures nearest to it (Morgan r2 / 2048 Tanimoto), i.e. engine-like hard decoys.
  V   the next N_V such molecules whose generic scaffold is not used by H1 / H2 / R (early stopping only).
  train  [M+H]+ spectra of every other structure whose key and generic Murcko scaffold (non-empty) differ from all of R, H1, H2, V,
      and that is not in enveda-np-examples: T_tims = Enveda-180 timsTOF, T_gen = other libraries' Orbitrap / Q-TOF spectra in eV.
Decoys for training (the official recipe's PubChem half, replaced by our local pool): up to 30 random same-formula pool structures per
training structure, excluding every evaluation key. The other half are mutation decoys, made on the fly by ms-pred's mutate().

Inputs (not in git): the spectrum cache and pool tables of casmi26-v2-pool / the fold-safe dataset under D:/casmi-data/directions/run/work,
generic scaffolds (casmi-work/frigidpilot/structs_scaf.parquet), MassSpecGym keys (D:/casmi-data/mist/ds/msgym/msgym_pool.parquet),
casmi-work/glacierft/r_meta.json, the tims items (casmi-work/tims/items_<reg>.json).
Out (casmi-work/glacierft/data/): split.json, train_tims.parquet, train_gen.parquet, structs.parquet, decoys.pkl,
items_H.json, items_V.json (gl_runner.py input format plus "truth": [bool per candidate]).
usage: python build_data.py [--n_h 400] [--n_v 200]
"""
import argparse, hashlib, json, os, pickle, re
import numpy as np, pandas as pd

WORK = os.environ.get('CACHE_WORK', 'D:/casmi-data/directions/run/work')
TRAIN = os.environ.get('TRAIN_PARQUET', 'D:/casmi-data/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet')
SCAF = 'C:/Users/hemre/casmi-work/frigidpilot/structs_scaf.parquet'
MSG = 'D:/casmi-data/mist/ds/msgym/msgym_pool.parquet'
GW = 'C:/Users/hemre/casmi-work/glacierft'
TIMS = 'C:/Users/hemre/casmi-work/tims'
SALT = 'glacierft-2026-10-18'
N_DEC = 40
LIB_E180, LIB_NPX = 0, 9
INSTR_TIMS, INSTR_ORBI, INSTR_TOF = 0, 1, 2


def _low():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def h(k):
    return hashlib.sha1(f'{SALT}|{k}'.encode()).hexdigest()


def fp(smi):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(m)


def nce(s):
    """= gl_runner._nce: mean of the numbers in collision_energy_orig (the runner's CE for NCE-unit spectra)."""
    v = [float(x) for x in re.findall(r'[0-9]+\.?[0-9]*', str(s))]
    return float(np.mean(v)) if v else np.nan


def ce_list(r):
    if r.ce_n >= 3:
        return [float(r.ce_min), float(r.ce_mean), float(r.ce_max)]
    if r.ce_n == 2:
        return [float(r.ce_min), float(r.ce_max)]
    return [float(r.ce_mean)]


def spectrum(row, meta, off, mz, it):
    a, b = int(off[row]), int(off[row + 1]); r = meta.loc[row]
    return dict(mz=[float(x) for x in mz[a:b]], it=[float(x) for x in it[a:b]], prec=float(r.precursor_mz), adduct='[M+H]+', mode=1,
                instrument='timsTOF', ce_ev=ce_list(r), ce_orig='', ce_units='eV', row=int(row))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n_h', type=int, default=400); ap.add_argument('--n_v', type=int, default=200)
    a = ap.parse_args(); _low()
    from rdkit import DataStructs, RDLogger
    RDLogger.DisableLog('rdApp.*')
    out = os.path.join(GW, 'data'); os.makedirs(out, exist_ok=True)
    meta = pd.read_parquet(os.path.join(WORK, 'spec_meta.parquet'))
    st = pd.read_parquet(os.path.join(WORK, 'train_structs.parquet'), columns=['sid', 'key', 'smiles', 'formula', 'n_heavy'])
    sc = pd.read_parquet(SCAF, columns=['sid', 'scaffold']).set_index('sid').scaffold
    st['scaffold'] = sc.reindex(st.sid).fillna('').values
    msg = set(pd.read_parquet(MSG, columns=['ik14_std']).ik14_std)
    pool = pd.read_parquet(os.path.join(WORK, 'pool_meta.parquet'), columns=['key', 'smiles', 'formula'])
    rm = json.load(open(os.path.join(GW, 'r_meta.json')))
    R_keys = {x[2] for x in rm['replay']}
    k2scaf = dict(zip(st.key, st.scaffold))
    npx_sids = set(meta.sid[meta.lib == LIB_NPX])
    # ---------------- H1: stack1 groups of replay timsTOF molecules not in MassSpecGym
    H1 = [g for g in rm['groups'] if g['ik14'] not in msg]
    # ---------------- H2 / V candidates: Enveda-180 timsTOF [M+H]+ molecules
    mh = meta[(meta.lib == LIB_E180) & (meta.instr == INSTR_TIMS) & (meta.adduct == '[M+H]+') & (meta['mode'] > 0) &
              (meta.ce_n.isin([1, 3])) & meta.ce_mean.notna() & (meta.n_clean >= 3)]
    rows_of = {k: list(v) for k, v in mh.groupby('sid').groups.items()}
    S = st.set_index('sid')
    cand = [sid for sid in rows_of if S.key[sid] not in msg and S.key[sid] not in R_keys and sid not in npx_sids]
    cand.sort(key=lambda s: h(S.key[s]))
    by_form = pool.groupby('formula').indices
    excl_scaf = {k2scaf.get(k, '') for k in R_keys} | {k2scaf.get(g['ik14'], '') for g in H1}
    excl_scaf.discard('')

    def group(sid):
        k, smi, f = S.key[sid], S.smiles[sid], S.formula[sid]
        idx = [i for i in by_form.get(f, []) if pool.key.values[i] != k]
        if not idx:
            return None
        tf = fp(smi)
        if tf is None:
            return None
        sims = []
        for i in idx:
            q = fp(pool.smiles.values[i])
            if q is not None:
                sims.append((DataStructs.TanimotoSimilarity(tf, q), pool.key.values[i], pool.smiles.values[i]))
        seen, dec = {k}, []
        for s, kk, smi_d in sorted(sims, key=lambda x: (-x[0], x[1])):
            if kk not in seen:
                seen.add(kk); dec.append(smi_d)
            if len(dec) >= N_DEC:
                break
        return [smi] + dec if dec else None

    H2, V = [], []
    h_scaf, v_scaf = set(), set()
    for sid in cand:
        if len(H2) >= a.n_h and len(V) >= a.n_v:
            break
        scf = S.scaffold[sid]
        if len(H2) < a.n_h:
            g = group(sid)
            if g:
                H2.append((sid, g)); h_scaf.add(scf)
            continue
        if scf and (scf in h_scaf or scf in excl_scaf):
            continue
        g = group(sid)
        if g:
            V.append((sid, g)); v_scaf.add(scf)
    eval_keys = R_keys | {g['ik14'] for g in H1} | {S.key[s] for s, _ in H2} | {S.key[s] for s, _ in V}
    eval_scaf = (excl_scaf | h_scaf | v_scaf) - {''}
    # ---------------- training structures and spectra
    ok = ~st.key.isin(eval_keys) & ~st.scaffold.isin(eval_scaf) & ~st.sid.isin(npx_sids)
    ok_sid = set(st.sid[ok])
    import pyarrow.parquet as pq
    orig = pq.read_table(TRAIN, columns=['collision_energy_orig']).column(0).to_pylist()
    assert len(orig) == len(meta)
    nce_rows = np.flatnonzero((meta.collision_energy_orig_units == 'NCE').values)
    meta['ce_run'] = meta.ce_mean.values                         # the runner's CE: mean(ce_ev), or the NCE numbers for NCE units
    meta.loc[nce_rows, 'ce_run'] = [nce(orig[i]) for i in nce_rows]
    pos = meta[(meta.adduct == '[M+H]+') & (meta['mode'] > 0) & meta.ce_run.notna() & (meta.n_clean >= 3) & meta.sid.isin(ok_sid)]
    t_tims = pos[(pos.lib == LIB_E180) & (pos.instr == INSTR_TIMS) & pos.ce_n.isin([1, 3])]
    t_gen = pos[~pos.lib.isin([LIB_E180, LIB_NPX]) & pos.instr.isin([INSTR_ORBI, INSTR_TOF])]
    for name, T in (('train_tims', t_tims), ('train_gen', t_gen)):
        d = pd.DataFrame(dict(row=T.index.values, sid=T.sid.values, prec=T.precursor_mz.values, ce=T.ce_run.values, lib=T.lib.values,
                              tok=np.where(T.instr.values == INSTR_ORBI, 'Orbitrap', 'QTOF')))
        d.to_parquet(os.path.join(out, name + '.parquet'), index=False)
    tr_sids = sorted(set(t_tims.sid) | set(t_gen.sid))
    S2 = st[st.sid.isin(tr_sids)][['sid', 'key', 'smiles', 'formula', 'n_heavy']]
    S2.to_parquet(os.path.join(out, 'structs.parquet'), index=False)
    rng = np.random.default_rng(20261018)
    pk = pool.key.values; ps = pool.smiles.values
    dec = {}
    for sid, k, f in zip(S2.sid, S2.key, S2.formula):
        idx = [i for i in by_form.get(f, []) if pk[i] != k and pk[i] not in eval_keys]
        if len(idx) > 30:
            idx = list(rng.choice(idx, 30, replace=False))
        dec[int(sid)] = [ps[i] for i in idx]
    pickle.dump(dec, open(os.path.join(out, 'decoys.pkl'), 'wb'), protocol=4)
    # ---------------- runner items
    off = np.load(os.path.join(WORK, 'spec_off.npy')); mz = np.load(os.path.join(WORK, 'spec_mz.npy'), mmap_mode='r')
    it = np.load(os.path.join(WORK, 'spec_it.npy'), mmap_mode='r')
    tims_items = {}
    for reg in ('mx', 'p2'):
        for x in json.load(open(f'{TIMS}/items_{reg}.json')):
            tims_items[(reg, x['mid'])] = x
    items_H = []
    for g in H1:
        x = tims_items[(g['reg'], g['mid'])]
        items_H.append(dict(mid=f"H1|{g['reg']}|{g['mid']}", set='H1', key=g['ik14'], cands=g['grp'], truth=g['tr'],
                            spectra=[{k: v for k, v in s.items() if k != 'si'} for s in x['spectra']]))
    for sid, g in H2:
        items_H.append(dict(mid=f'H2|{S.key[sid]}', set='H2', key=S.key[sid], cands=g, truth=[True] + [False] * (len(g) - 1),
                            spectra=[spectrum(r, meta, off, mz, it) for r in rows_of[sid]]))
    items_V = [dict(mid=f'V|{S.key[sid]}', set='V', key=S.key[sid], cands=g, truth=[True] + [False] * (len(g) - 1),
                    spectra=[spectrum(r, meta, off, mz, it) for r in rows_of[sid]]) for sid, g in V]
    json.dump(items_H, open(os.path.join(out, 'items_H.json'), 'w'))
    json.dump(items_V, open(os.path.join(out, 'items_V.json'), 'w'))
    info =dict(salt=SALT, n_H1=len(H1), n_H2=len(H2), n_V=len(V), R=len(R_keys), eval_keys=len(eval_keys), eval_scaffolds=len(eval_scaf),
                train_structs=len(S2), train_tims_spectra=len(t_tims), train_tims_structs=int(t_tims.sid.nunique()),
                train_gen_spectra=len(t_gen), train_gen_structs=int(t_gen.sid.nunique()),
                train_gen_libs={int(k): int(v) for k, v in t_gen.groupby('lib').sid.nunique().items()},
                excluded_structs_by_scaffold=int((st.scaffold.isin(eval_scaf)).sum()),
                e180_mh_structs=len(rows_of), group_size_H2=float(np.median([len(g) for _, g in H2])),
                group_size_V=float(np.median([len(g) for _, g in V])), group_size_H1=float(np.median([len(g['grp']) for g in H1])),
                spectra_per_mol_H2=float(np.mean([len(rows_of[s]) for s, _ in H2])), decoys_per_struct=float(np.mean([len(v) for v in dec.values()])))
    json.dump(dict(info=info, H1=[x['mid'] for x in items_H if x['set'] == 'H1'], H2=[S.key[s] for s, _ in H2], V=[S.key[s] for s, _ in V],
                   eval_keys=sorted(eval_keys), eval_scaffolds=sorted(eval_scaf)), open(os.path.join(out, 'split.json'), 'w'), indent=0)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
