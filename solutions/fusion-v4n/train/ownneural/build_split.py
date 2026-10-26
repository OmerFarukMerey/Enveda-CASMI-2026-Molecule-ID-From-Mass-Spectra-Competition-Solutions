"""Training / validation / excluded roles of every training structure for the ownneural model (scaffold-disjoint from the evaluation molecules).

Roles (one per structure of train_structs.parquet, written to $CASMI_DATA/ownneural/split.parquet):
  eval      an answer of the evaluation dumps (s1val, s1b): same InChIKey first block, metric key or raw key
  eval_scaf shares a non-empty Murcko scaffold with an `eval` structure
  H         a molecule of the clean chain's frozen held-out set (file given with --h; identity match), and H_scaf (its scaffold, plus every
            structure listed in the clean chain's exclusion file --e, column sid)
  val       Enveda validation: whole scaffold groups (size <= --max_group) of structures with Enveda spectra, drawn with --seed
  train     everything else (the positives)
Acyclic structures (empty scaffold) are grouped by their own metric key. Excluded structures are neither positives nor pool negatives.

usage: python build_split.py [--h <file with key / inchikey14 column>] [--e <file with sid column>] [--n_val 1500] [--seed 0] [--force]
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onpaths as P  # noqa: E402

sys.path.insert(0, P.CLEAN)


def answer_ids(smiles, ik14):
    """Identifiers of the evaluation answers: label InChIKey14, plus metric key and raw key of the standardised structure."""
    import pool_build
    ids = set(x for x in ik14 if isinstance(x, str))
    for s in smiles:
        st = pool_build.standardise(s)
        rc = pool_build.record(st[0]) if st else None
        if rc is not None:
            ids.add(rc[0]['key']); ids.add(rc[0]['rawkey'])
    return ids


def identity_mask(st, ids):
    m = np.zeros(len(st), bool)
    for c in ('inchikey14', 'key', 'rawkey'):
        m |= st[c].isin(ids).values
    return m


def scaffold_groups(st, scaf):
    """Group label per structure: its Murcko scaffold, or 'k:' + metric key when acyclic / failed."""
    s = scaf.reindex(st.sid.values).values
    return np.where(pd.isna(s) | (s == ''), 'k:' + st.key.values.astype(str), s.astype(object)).astype(str)


def assign(st, grp, eval_ids, h_ids, env, n_val, max_group, seed, e_sids=()):
    """Role array. st: train_structs (sid order); grp: scaffold group per structure; env: has Enveda spectra."""
    role = np.full(len(st), 'train', object)
    ev = identity_mask(st, eval_ids); ev_g = set(grp[ev])
    role[np.isin(grp, list(ev_g)) & ~ev] = 'eval_scaf'; role[ev] = 'eval'
    if h_ids:
        h = identity_mask(st, h_ids) & (role == 'train'); h_g = set(grp[h])
        role[np.isin(grp, list(h_g)) & ~h & (role == 'train')] = 'H_scaf'; role[h] = 'H'
    if len(e_sids):
        role[np.isin(st.sid.values, np.asarray(list(e_sids))) & (role == 'train')] = 'H_scaf'
    free = role == 'train'
    g = pd.DataFrame(dict(g=grp[free], env=env[free])).groupby('g').agg(n=('env', 'size'), e=('env', 'sum'))
    g = g[(g.n <= max_group) & (g.e == g.n)]                                   # whole groups, every member with Enveda spectra
    order = np.random.default_rng(seed).permutation(len(g)); take, tot = [], 0
    for i in order:
        if tot >= n_val:
            break
        take.append(g.index[i]); tot += int(g.n.iloc[i])
    role[free & np.isin(grp, take)] = 'val'
    return role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', default=''); ap.add_argument('--e', default=''); ap.add_argument('--n_val', type=int, default=1500); ap.add_argument('--max_group', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--force', action='store_true')
    a = ap.parse_args(); os.makedirs(P.ON, exist_ok=True); out = P.ON + '/split.parquet'
    if os.path.exists(out) and not a.force:
        print('exists', out); return
    st = pd.read_parquet(P.WORK + '/train_structs.parquet'); assert (st.sid.values == np.arange(len(st))).all()
    sm = pd.read_parquet(P.WORK + '/spec_meta.parquet', columns=['sid', 'lib'])
    env = np.zeros(len(st), bool); env[sm.sid.values[sm.lib.isin([0, 9]).values]] = True
    sc = pd.read_parquet(P.SCAFFOLDS).set_index('sid').scaffold; assert len(sc) == len(st)
    grp = scaffold_groups(st, sc)
    labs = pd.concat([pd.read_csv(d + '/s1val_labels.csv').assign(dump=k) for k, d in P.DUMPS.items()])
    eval_ids = answer_ids(labs.smiles.unique(), labs.inchikey14.unique())
    h_ids = set()
    if a.h:
        H = pd.read_parquet(a.h) if a.h.endswith('.parquet') else pd.read_csv(a.h)
        for c in ('key', 'inchikey14', 'ik14', 'rawkey'):
            if c in H:
                h_ids |= set(H[c].dropna().astype(str))
        assert h_ids, 'no identifier column in ' + a.h
    e_sids = set(pd.read_parquet(a.e).sid.astype(int)) if a.e else set()
    role = assign(st, grp, eval_ids, h_ids, env, a.n_val, a.max_group, a.seed, e_sids)
    S = pd.DataFrame(dict(sid=st.sid.values, role=role.astype(str), group=grp, env=env))
    S.to_parquet(out, index=False)
    found = labs.inchikey14.isin(set(st.inchikey14[S.role == 'eval'])) | labs.inchikey14.isin(set(st.key[S.role == 'eval']))
    stats = dict(roles=S.role.value_counts().to_dict(), env_by_role=S[S.env].role.value_counts().to_dict(), eval_labels=int(labs.inchikey14.nunique()),
                 eval_labels_found_in_train=int(labs[found].inchikey14.nunique()), eval_ids=len(eval_ids), h_file=a.h, h_ids=len(h_ids), e_file=a.e, e_sids=len(e_sids), seed=a.seed, n_val=a.n_val)
    json.dump(stats, open(P.ON + '/split_stats.json', 'w'), indent=1); print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
