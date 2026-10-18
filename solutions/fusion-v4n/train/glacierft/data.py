"""GLACIER fine-tune samples: one training spectrum + its decoys, featurised with ms-pred's own TreeProcessor, batched by ms-pred's
IntenContrDataset.collate_fn (the contrastive fine-tune's batch layout: {"targ", "decoy", "num_decoys_per_entry"}).

Target (= ms-pred's MSG pipeline for `raw_spec`): cache peaks -> common.process_spec_file (peaks <= precursor + 1, max-normalised,
sqrt) -> common.max_inten_spec(50 peaks, intensity > 0.001). Tree = {root_canonical_smiles, adduct, collision_energy, frags: {},
raw_spec}; no MAGMa fragments (the fragment loss is off in this fine-tune, README).
Conditioning = what stack1's runner (gl_runner.py) feeds at inference: adduct [M+H]+, collision energy = the runner's 5-eV bucket of the
spectrum's CE (mean of ce_ev; NCE numbers for NCE units), instrument token Orbitrap for Orbitrap / FT, QTOF otherwise (timsTOF -> QTOF).
Decoys (= IntenContrDataset.generate_decoys with the PubChem HDF5 lookup replaced by our pool list): all_decoy_nums candidates, half
ms-pred mutate() mutations of the target, half random same-formula pool structures (build_data.py decoys.pkl); stereo removed, sanitised,
unique by InChIKey, no '.', not the target; num_decoys of them kept at random.
"""
import os, pickle, random
import numpy as np
import pandas as pd

WORK = os.environ.get('CACHE_WORK', 'D:/casmi-data/directions/run/work')


def ce_bucket(ce):
    """= gl_runner.ce_bucket."""
    ceb = float(np.round(ce / 5.0) * 5.0) if np.isfinite(ce) else np.nan
    if np.isfinite(ceb) and ceb < 5:
        ceb = 5.0
    return ceb


class Samples:
    """Training table (train_tims / train_gen rows) + structures + decoy lists; the per-sample featurisation runs in DataLoader workers."""

    def __init__(self, data_dir, num_decoys=5, all_decoy_nums=20, max_heavy=64):
        self.data_dir, self.num_decoys, self.all_decoy_nums, self.max_heavy = data_dir, num_decoys, all_decoy_nums, max_heavy
        st = pd.read_parquet(os.path.join(data_dir, 'structs.parquet'))
        st = st[st.n_heavy <= max_heavy]
        self.smiles = dict(zip(st.sid, st.smiles))
        self.tables = {}
        for name in ('tims', 'gen'):
            t = pd.read_parquet(os.path.join(data_dir, f'train_{name}.parquet'))
            t = t[t.sid.isin(self.smiles.keys())].reset_index(drop=True)
            self.tables[name] = t
        self.by_sid = {name: t.groupby('sid').indices for name, t in self.tables.items()}
        self.sids = {name: np.array(sorted(v.keys())) for name, v in self.by_sid.items()}
        self.decoys = pickle.load(open(os.path.join(data_dir, 'decoys.pkl'), 'rb'))
        self._mm = None; self._tp = None

    # ---------------------------------------------------------------- sampling (main process, deterministic per epoch)
    def epoch(self, e, n, p_tims, seed):
        """n (table, row index) pairs: structure-uniform within each table, a random spectrum of that structure."""
        rng = np.random.default_rng([seed, e])
        out = []
        for _ in range(n):
            name = 'tims' if rng.random() < p_tims else 'gen'
            sid = self.sids[name][rng.integers(len(self.sids[name]))]
            idx = self.by_sid[name][sid]
            out.append((name, int(idx[rng.integers(len(idx))])))
        return out

    # ---------------------------------------------------------------- featurisation (worker)
    def _init(self):
        if self._tp is None:
            import env
            self.common = env.setup()
            from ms_pred.glacier import dataset as gd
            from ms_pred.iceberg.graph_mutate import mutate
            self.gd, self.mutate = gd, mutate
            self._tp = gd.TreeProcessor(pe_embed_k=14, root_encode='graphormer', embed_elem_group=True, multi_hop_max_dist=10)
            self._mm = (np.load(os.path.join(WORK, 'spec_off.npy')), np.load(os.path.join(WORK, 'spec_mz.npy'), mmap_mode='r'),
                        np.load(os.path.join(WORK, 'spec_it.npy'), mmap_mode='r'))

    def target_spec(self, row, prec):
        off, mz, it = self._mm
        a, b = int(off[row]), int(off[row + 1])
        arr = np.stack([np.asarray(mz[a:b], np.float64), np.asarray(it[a:b], np.float64)], 1)
        sp = self.common.process_spec_file({'parentmass': prec}, [('collision 0', arr)], merge_specs=False)
        if not sp:
            return None
        spec = self.common.max_inten_spec(sp['collision 0'], 50, inten_thresh=0.001)
        return spec if len(spec) else None

    def canon(self, smi):
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        for atom in m.GetAtoms():
            if atom.GetAtomicNum() not in self.common.chem_utils.VALID_ATOM_NUM:
                return None
        m = Chem.RemoveHs(m)
        return Chem.MolToSmiles(m) if 0 < m.GetNumAtoms() <= 100 else None

    def gen_decoys(self, smi, sid):
        from rdkit import Chem
        common = self.common
        smi = common.rm_stereo(smi)
        mol = common.smi_inchi_round_mol(smi)
        ik = Chem.MolToInchiKey(mol)
        pool = self.decoys.get(int(sid), [])
        n_pool = self.all_decoy_nums // 2 if pool else 0
        mols = [self.mutate(Chem.Mol(mol), mutation_rate=1.) for _ in range(self.all_decoy_nums - n_pool)]
        if n_pool:
            mols += [Chem.MolFromSmiles(s) for s in random.choices(pool, k=n_pool)]
        mols = [common.rm_stereo(m, mol_type='mol') for m in mols if m is not None]
        mols = np.array(common.sanitize(mols))
        if len(mols) == 0:
            return []
        iks = np.array([Chem.MolToInchiKey(m) for m in mols])
        _, u = np.unique(iks, return_index=True)
        out = []
        for m, k in zip(mols[u], iks[u]):
            s = Chem.MolToSmiles(m)
            if '.' not in s and k != ik:
                c = self.canon(s)
                if c:
                    out.append(c)
        return random.sample(out, self.num_decoys) if len(out) > self.num_decoys else out

    def item(self, smi_c, ce, tok, prec, spec=None):
        tree = {'root_canonical_smiles': smi_c, 'adduct': '[M+H]+', 'collision_energy': ce}
        if spec is not None:
            tree.update(frags={}, raw_spec=spec)
        f = self._tp.featurize_tree(tree, include_inten_targs=spec is not None, include_frag_targs=spec is not None)
        d = {'name': smi_c, 'adduct': self.common.ion2onehot_pos['[M+H]+'], 'precursor': float(prec),
             'instrument': self.common.instrument2onehot_pos[tok]}
        d.update(f)
        d['collision_energy'] = float(ce)
        return d

    def sample(self, name, i):
        """-> [target item, [decoy items]] or None (unusable spectrum / structure)."""
        self._init()
        r = self.tables[name].iloc[i]
        spec = self.target_spec(int(r.row), float(r.prec))
        smi_c = self.canon(self.smiles[int(r.sid)])
        ce = ce_bucket(float(r.ce))
        if spec is None or smi_c is None or not np.isfinite(ce):
            return None
        try:
            t = self.item(smi_c, ce, r.tok, r.prec, spec)
            decs = []
            for d in self.gen_decoys(smi_c, r.sid):
                try:
                    decs.append(self.item(d, ce, r.tok, r.prec))
                except Exception:
                    pass
        except Exception:
            return None
        return [t, decs] if decs else None


class EpochData:
    """torch map-style dataset over one epoch's (table, row) list from position `start`."""

    def __init__(self, samples, pairs, start=0):
        self.s, self.pairs, self.start = samples, pairs, start

    def __len__(self):
        return len(self.pairs) - self.start

    def __getitem__(self, k):
        name, i = self.pairs[self.start + k]
        return self.s.sample(name, i)


def collate(batch):
    """ms-pred IntenContrDataset.collate_fn over the usable samples (None dropped)."""
    import env
    env.setup()
    from ms_pred.glacier import dataset as gd
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return gd.IntenContrDataset.collate_fn(batch), len(batch)
