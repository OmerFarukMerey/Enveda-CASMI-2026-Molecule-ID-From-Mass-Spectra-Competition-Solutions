"""CASMI26 spectrum-candidate scorer, retrained longer and wider on a scaffold-disjoint holdout.

Derived from sanpier/casmi-spectrum-model-v1 (Apache-2.0: features.py, model.py, prepare.py, train.py); this script adds
parallel structure preparation, a scaffold-disjoint holdout that also holds out the Enveda NP example molecules, a
warm-up + cosine learning-rate schedule, the competition adduct names, and wider models. Runs S (the published recipe,
the baseline), L (the published model, 7.5x the steps) and W (wider, adduct names), then refits the better of L / W on
every identity. Only train.parquet is read; the visible test file is never opened.
"""
import os, sys, json, time, math, hashlib, subprocess, collections, random
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES_SRC = '"""Shared training/inference features (sanpier/casmi-spectrum-model-v1, Apache-2.0) with the competition\'s adduct names added."""\nimport ast, json\nimport numpy as np\n# Indices 1-10 are the original vocabulary; 11-14 are competition adducts it mapped to 0 (unknown).\nADDUCTS = [\'[M+H]+\',\'[M-H]-\',\'[M+Na]+\',\'[M+K]+\',\'[M+NH4]+\',\'[M+Cl]-\',\'[M+FA-H]-\',\'[M+H-H2O]+\',\'[M+2H]2+\',\'[M-2H]2-\',\n           \'[M-H2O+H]+\',\'[M-2H2O+H]+\',\'[M-H2O-H]-\',\'[M+CH2O2-H]-\']\ndef array_value(x):\n    if isinstance(x,str):\n        try: x=json.loads(x)\n        except (ValueError,TypeError):\n            try:x=ast.literal_eval(x)\n            except (ValueError,SyntaxError):x=x.replace(\',\',\' \').split()\n    return np.asarray(x,dtype=np.float32).reshape(-1)\ndef peaks(mzs,intensities,precursor,max_peaks=96):\n    mz,it=array_value(mzs),array_value(intensities)\n    if len(mz)!=len(it): raise ValueError(\'Peak and intensity lengths differ\')\n    keep=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0)\n    mz,it=mz[keep],it[keep]\n    if not len(mz):raise ValueError(\'No valid peaks\')\n    ix=np.argsort(-it,kind=\'stable\')[:max_peaks];ix=ix[np.argsort(mz[ix],kind=\'stable\')]\n    mz,it=mz[ix],it[ix];it=it/it.max()\n    out=np.zeros((max_peaks,4),np.float32)\n    out[:len(mz)]=np.stack([mz/1000,np.sqrt(it),np.log1p(100*it)/np.log(101), (float(precursor)-mz)/1000],1)\n    return out,len(mz)\ndef adduct_id(adduct,vocab=None):\n    vocab=ADDUCTS if vocab is None else vocab\n    return vocab.index(adduct)+1 if adduct in vocab else 0\ndef metadata(precursor,adduct,mode,ce):\n    try:\n        a=array_value(ce);a=a[np.isfinite(a)];ev=float(a.mean()) if len(a) else 0.;known=float(len(a)>0)\n    except (ValueError,TypeError):ev,known=0.,0.\n    mode=str(mode).lower();pol=1. if mode in (\'positive\',\'pos\',\'1\',\'+\') else -1. if mode in (\'negative\',\'neg\',\'-1\',\'-\') else (1. if str(adduct).endswith(\'+\') else -1. if str(adduct).endswith(\'-\') else 0.)\n    return np.array([float(precursor)/1000,np.clip(ev,0,500)/100,known,pol],np.float32),adduct_id(adduct)\n\ndef structure(smiles,nbits=2048):\n    from rdkit import Chem\n    from rdkit.Chem import rdFingerprintGenerator,rdMolDescriptors\n    from rdkit.Chem.MolStandardize import rdMolStandardize\n    mol=Chem.MolFromSmiles(str(smiles))\n    if mol is None:raise ValueError(\'Invalid structure\')\n    Chem.RemoveStereochemistry(mol)\n    mol=rdMolStandardize.TautomerEnumerator().Canonicalize(mol)\n    key=Chem.MolToInchiKey(mol)[:14]\n    if not key:raise ValueError(\'No InChIKey\')\n    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=nbits).GetFingerprintAsNumPy(mol)\n    return key,Chem.MolToSmiles(mol),rdMolDescriptors.CalcMolFormula(mol),rdMolDescriptors.CalcExactMolWt(mol),fp\n'
MODEL_SRC = '"""Spectrum-candidate bi-encoder (sanpier/casmi-spectrum-model-v1, Apache-2.0); config \'members\' makes it an ensemble.\n\nAn ensemble concatenates the members\' unit embeddings scaled by 1/sqrt(K), so z @ m equals the mean member cosine,\nand averages the fingerprint logits; every caller of the single model gets the mean member score unchanged.\n"""\nimport math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nclass SingleModel(nn.Module):\n    def __init__(self,cfg):\n        super().__init__();d=cfg[\'width\'];self.cfg=cfg\n        self.register_buffer(\'freq\',2.**torch.arange(10).float())\n        self.peak=nn.Linear(44,d)\n        self.meta=nn.Sequential(nn.Linear(4,d),nn.GELU(),nn.Linear(d,d))\n        self.adduct=nn.Embedding(cfg.get(\'n_adducts\',11),d);self.cls=nn.Parameter(torch.zeros(1,1,d))\n        layer=nn.TransformerEncoderLayer(d,cfg[\'heads\'],d*4,cfg[\'dropout\'],batch_first=True,norm_first=True)\n        self.transformer=nn.TransformerEncoder(layer,cfg[\'layers\'],enable_nested_tensor=False)\n        self.norm=nn.LayerNorm(d)\n        self.spec_projection=nn.Linear(d,cfg[\'embedding\'])\n        self.fp_head=nn.Linear(d,cfg[\'fp_bits\'])\n        self.molecule=nn.Sequential(nn.Linear(cfg[\'fp_bits\'],cfg.get(\'mol_hidden\',512)),nn.GELU(),nn.Dropout(cfg[\'dropout\']),nn.Linear(cfg.get(\'mol_hidden\',512),cfg[\'embedding\']))\n    def encode_spectra(self,x,lengths,meta,adduct):\n        # A member trained on a smaller adduct vocabulary sees the newer adducts as unknown (0), as in its training.\n        adduct=adduct.masked_fill(adduct>=self.adduct.num_embeddings,0)\n        a=x[:,:,[0,3]].unsqueeze(-1)*self.freq*torch.pi\n        p=torch.cat([x,a.sin().flatten(2),a.cos().flatten(2)],-1)\n        tokens=self.peak(p)\n        cls=self.cls.expand(len(x),-1,-1)+self.meta(meta)[:,None]+self.adduct(adduct)[:,None]\n        tokens=torch.cat([cls,tokens],1)\n        mask=torch.arange(x.shape[1]+1,device=x.device)[None]>lengths[:,None]\n        h=self.norm(self.transformer(tokens,src_key_padding_mask=mask)[:,0])\n        return F.normalize(self.spec_projection(h),dim=-1),self.fp_head(h)\n    def encode_molecules(self,fp):return F.normalize(self.molecule(fp),dim=-1)\n\nclass SpectrumCandidateModel(nn.Module):\n    def __init__(self,cfg):\n        super().__init__();self.cfg=cfg\n        members=cfg.get(\'members\') or [{}]\n        self.members=nn.ModuleList([SingleModel({**cfg,**m}) for m in members]);self.scale=1/math.sqrt(len(members))\n    def encode_spectra(self,x,lengths,meta,adduct):\n        out=[m.encode_spectra(x,lengths,meta,adduct) for m in self.members]\n        return torch.cat([z for z,_ in out],-1)*self.scale,torch.stack([p for _,p in out]).mean(0)\n    def encode_molecules(self,fp):return torch.cat([m.encode_molecules(fp) for m in self.members],-1)*self.scale\n'

T0 = time.time()
LOCAL = os.environ.get('NN_LOCAL_TEST') == '1'
WORK = Path(os.environ.get('NN_WORK', '/kaggle/working'))
INP = Path(os.environ.get('NN_INPUT', '/kaggle/input'))
SRC = WORK / 'nn_src'
CACHE = WORK / 'cache'
OUT = WORK / 'out'
SEED = 20261023


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def _pool_init(src):
    global FT
    sys.path.insert(0, src)
    import features as FT  # noqa: F811


def struct_job(s):
    """One unique SMILES -> (key, canonical, formula, mass, packed fp, Murcko scaffold) or None (rejected as in the kit)."""
    try:
        key, can, formula, mass, fp = FT.structure(s, 2048)
    except (ValueError, RuntimeError):
        return s, None
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(can))
    except Exception:
        scaf = ''
    return s, (key, can, formula, float(mass), np.packbits(fp.astype(np.uint8)), scaf)


def prepare(train_path, workers):
    """Kit prepare.py with the same filters and dedup; structures in parallel; adduct and library kept as strings."""
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / 'done.json').exists():
        log('PREP cached'); return json.loads((CACHE / 'done.json').read_text())
    pf = pq.ParquetFile(train_path)
    names = pf.schema_arrow.names
    scol = next(c for c in ['smiles', 'canonical_smiles', 'normalized_smiles'] if c in names)
    uniq = pd.unique(pq.read_table(train_path, columns=[scol]).column(0).to_pandas().astype(str))
    log('PREP unique smiles', len(uniq), 'column', scol)
    cache = {}
    with Pool(workers, initializer=_pool_init, initargs=(str(SRC),)) as pool:
        for n, (s, st) in enumerate(pool.imap_unordered(struct_job, uniq, chunksize=256), 1):
            cache[s] = st
            if n % 50000 == 0: log('PREP structures', n)
    log('PREP structures done', len(cache), 'invalid', sum(v is None for v in cache.values()))
    reject = collections.Counter(); molecules = {}; rows = []; seen = set(); total = 0
    px = open(CACHE / 'peaks.bin', 'wb'); mx = open(CACHE / 'metadata.bin', 'wb')
    cols = [scol, 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct', 'ionization_mode',
            'collision_energy_ev', 'ingest_lib', 'instrument_type']
    cols = [c for c in cols if c in names]
    try:
        for b in pf.iter_batches(batch_size=8192, columns=cols):
            frame = b.to_pandas()
            for row in frame.to_dict('records'):
                total += 1
                try:
                    st = cache[str(row[scol])]
                    if st is None: reject['invalid_structure'] += 1; continue
                    key, can, formula, mass, packed, scaf = st
                    precursor = float(row['precursor_mz'])
                    if not np.isfinite(precursor) or precursor <= 0: reject['invalid_precursor'] += 1; continue
                    pk, length = FT.peaks(row['ms2_mzs'], row['ms2_normalized_intensities'], precursor, 96)
                    if length < 3: reject['too_few_peaks'] += 1; continue
                    ad = str(row.get('adduct', '') or ''); mo = row.get('ionization_mode', ''); ce = row.get('collision_energy_ev', None)
                    mt, _ = FT.metadata(precursor, ad, mo, ce)
                    digest = hashlib.sha256(key.encode() + pk.astype('<f2').tobytes() + mt.astype('<f4').tobytes() + ad.encode()).digest()
                    if digest in seen: reject['duplicate_model_input'] += 1; continue
                    seen.add(digest)
                    if key not in molecules: molecules[key] = (can, formula, mass, packed, scaf)
                    pk.astype('<f2').tofile(px); mt.astype('<f4').tofile(mx)
                    rows.append((key, length, ad, str(row.get('ingest_lib', '')), str(row.get('instrument_type', ''))))
                except (ValueError, TypeError, OverflowError):
                    reject['invalid_spectrum'] += 1
            if total % 409600 < 8192: log('PREP rows', total, 'accepted', len(rows))
    finally:
        px.close(); mx.close()
    keys = sorted(molecules); lookup = {k: i for i, k in enumerate(keys)}
    mol = pd.DataFrame([(k, *molecules[k][:3], molecules[k][4]) for k in keys], columns=['key', 'smiles', 'formula', 'mass', 'scaffold'])
    fps = np.unpackbits(np.stack([molecules[k][3] for k in keys]), axis=1)[:, :2048].astype(np.uint8)
    np.save(CACHE / 'fingerprints.npy', fps)
    spec = pd.DataFrame(rows, columns=['key', 'length', 'adduct', 'lib', 'instrument'])
    spec['mol_idx'] = spec.key.map(lookup).astype(np.int64)
    np_keys = set(spec.loc[spec.lib.eq('enveda-np-examples'), 'key'])
    mol['np_example'] = mol.key.isin(np_keys)
    mol['libs'] = mol.key.map(spec.groupby('key').lib.agg(lambda s: '|'.join(sorted(set(s)))))
    mol.to_parquet(CACHE / 'molecules.parquet', index=False); spec.to_parquet(CACHE / 'spectra.parquet', index=False)
    report = dict(input_rows=total, accepted=len(spec), molecules=len(mol), rejected=dict(reject), np_examples=len(np_keys),
                  adducts=spec.adduct.value_counts().to_dict(), seconds=round(time.time() - T0))
    (CACHE / 'done.json').write_text(json.dumps(report, indent=1)); log('PREP', json.dumps(report)[:1500])
    return report


def scaffold_split(mol, frac=0.10, cap=200, np_cap=500):
    """Whole Murcko groups (acyclic = singleton) in a seeded hash order until frac; groups above cap always train.
    Every NP example's group goes to validation when it has at most np_cap members; otherwise the example alone (loose)."""
    group = np.where(mol.scaffold.eq(''), 'ACYCLIC:' + mol.key, mol.scaffold)
    sizes = pd.Series(group).value_counts()
    val = np.zeros(len(mol), bool); loose = np.zeros(len(mol), bool)
    npg = set(group[mol.np_example.values])
    for g in npg:
        idx = group == g
        if sizes[g] <= np_cap: val |= idx
    np_left = mol.np_example.values & ~val
    val |= np_left; loose |= np_left
    order = sorted((g for g in sizes.index if sizes[g] <= cap and g not in npg),
                   key=lambda g: hashlib.sha256((str(SEED) + g).encode()).hexdigest())
    target = int(frac * len(mol)); members = collections.defaultdict(list)
    for i, g in enumerate(group): members[g].append(i)
    for g in order:
        if val.sum() >= target: break
        val[members[g]] = True
    split = np.where(val, 'validation', 'train')
    shared = set(group[val & ~loose]) & set(group[~val])
    assert not shared, f'{len(shared)} scaffolds on both sides'
    return split, loose


class Data:
    def __init__(self):
        self.mol = pd.read_parquet(CACHE / 'molecules.parquet'); self.spec = pd.read_parquet(CACHE / 'spectra.parquet')
        n = len(self.spec)
        self.x = np.memmap(CACHE / 'peaks.bin', mode='r', dtype='<f2', shape=(n, 96, 4))
        self.meta = np.memmap(CACHE / 'metadata.bin', mode='r', dtype='<f4', shape=(n, 4))
        self.fp = np.load(CACHE / 'fingerprints.npy', mmap_mode='r')
        self.lengths = self.spec.length.values
        self.by_mol = {int(i): g.index.to_numpy() for i, g in self.spec.groupby('mol_idx', sort=False)}
        split, loose = scaffold_split(self.mol)
        self.mol['split'] = split; self.mol['loose'] = loose
        self.train = np.where(split == 'train')[0]; self.val = np.where(split == 'validation')[0]
        self._ad = {}

    def adducts(self, vocab):
        if vocab not in self._ad:
            names = FT.ADDUCTS[:10] if vocab == 'old' else FT.ADDUCTS
            lut = {a: i + 1 for i, a in enumerate(names)}
            self._ad[vocab] = self.spec.adduct.map(lambda a: lut.get(a, 0)).values.astype(np.int64)
        return self._ad[vocab]

    def batch(self, rows, adducts, device, augment=False):
        return self.batch_np(np.asarray(self.x[rows]), self.lengths[rows], np.asarray(self.meta[rows]), adducts[rows], device, augment)

    def batch_np(self, xs, lengths, meta, adducts, device, augment=False):
        import torch
        x = torch.tensor(xs, device=device, dtype=torch.float32)
        if augment:
            it = (x[:, :, 1].square() * torch.exp(.08 * torch.randn_like(x[:, :, 1]))).clamp_min(0)
            it = it / it.max(dim=1, keepdim=True).values.clamp_min(1e-8)
            x[:, :, 1] = it.sqrt(); x[:, :, 2] = torch.log1p(100 * it) / np.log(101)
        return (x, torch.tensor(lengths, device=device, dtype=torch.long),
                torch.tensor(meta, device=device), torch.tensor(adducts, device=device, dtype=torch.long))


def hard_pools(mol, allowed, count, only=None):
    """Kit hard_pools: same-formula identities first, then the nearest masses; restricted to `allowed`."""
    allowed = np.asarray(allowed); groups = collections.defaultdict(list)
    formulas = mol.formula.values; mass_values = mol.mass.values
    for i in allowed: groups[formulas[i]].append(int(i))
    order = allowed[np.argsort(mass_values[allowed], kind='stable')]; masses = mass_values[order]; pools = {}
    for i in (allowed if only is None else only):
        same = [j for j in groups[formulas[i]] if j != i]
        pos = int(np.searchsorted(masses, mass_values[i]))
        nearby = order[max(0, pos - count):min(len(order), pos + count + 1)].tolist()
        nearby.sort(key=lambda j: (abs(mass_values[j] - mass_values[i]), j))
        seen = {int(i)}; out = []
        for j in same + nearby:
            if j not in seen: seen.add(j); out.append(j)
            if len(out) >= count: break
        pools[int(i)] = np.asarray(out, np.int64)
    return pools


def diagnostic_set(data, n_total):
    rng = np.random.default_rng(SEED)
    npv = data.val[data.mol.np_example.values[data.val]]
    rest = np.setdiff1d(data.val, npv)
    pick = np.r_[npv, rng.choice(rest, min(len(rest), max(0, n_total - len(npv))), replace=False)]
    pools = hard_pools(data.mol, np.arange(len(data.mol)), 255, only=pick)
    return {int(i): np.r_[i, pools[int(i)]] for i in pick}


def validate(model, data, cands, adducts, cfg, device):
    import torch
    model.eval(); rows = []; formula = data.mol.formula.values
    with torch.inference_mode():
        for i, ids in cands.items():
            sr = data.by_mol[i]; zs = 0; ps = 0; n = 0
            for st in range(0, len(sr), 256):
                b = sr[st:st + 256]
                z, p = model.encode_spectra(*data.batch(b, adducts, device)); zs = zs + z.float().sum(0); ps = ps + p.float().sum(0); n += len(b)
            fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
            s = ((model.encode_molecules(fp).float() @ (zs / n)) / cfg['temperature'] + cfg['fp_score_weight'] * (fp @ (ps / n)) / cfg['fp_bits']).cpu().numpy()
            rank = 1 + int(np.sum(s[1:] >= s[0]))
            same = formula[ids[1:]] == formula[i]
            frank = 1 + int(np.sum(s[1:][same] >= s[0]))
            rows.append(dict(i=i, rank=rank, rr=1 / rank if rank <= 25 else 0., top1=int(rank == 1), frank=frank,
                             frr=1 / frank, n_same=int(same.sum()), np=bool(data.mol.np_example.values[i]), spectra=n))
    f = pd.DataFrame(rows)
    res = dict(mrr25=float(f.rr.mean()), top1=float(f.top1.mean()), iso_mrr=float(f.frr.mean()),
               np_mrr25=float(f[f.np].rr.mean()) if f.np.any() else None, np_iso_mrr=float(f[f.np].frr.mean()) if f.np.any() else None,
               n=len(f), n_np=int(f.np.sum()))
    model.train(); return res, f


def train_run(name, data, cfg, ids, cands, epochs, val_every, device, deadline, seed):
    import torch
    from torch.nn import functional as F
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    adducts = data.adducts(cfg['vocab'])
    t = time.time(); pools = hard_pools(data.mol, ids, cfg['negative_pool']); log(name, 'pools', len(pools), round(time.time() - t), 's')
    model = MD.SingleModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    spe = cfg['steps_per_epoch']; total = epochs * spe; warm = min(cfg.get('warmup', 0), total // 5)
    floor = cfg.get('lr_floor', 0.05)
    def lr_at(s):
        if not cfg.get('cosine'): return 1.0
        if s < warm: return (s + 1) / warm
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = device.type == 'cuda'; scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    counts = np.zeros(cfg['fp_bits'], np.float64)
    for st in range(0, len(ids), 4096): counts += np.asarray(data.fp[ids[st:st + 4096]]).sum(0)
    pos_weight = torch.tensor(np.clip((len(ids) - counts) / (counts + 1), 1, 20), device=device, dtype=torch.float32)
    best = -1.; history = []; stopped = False; B = min(cfg['batch_size'], len(ids)); K = cfg['hard_negatives']
    # Fingerprints stay on the device as uint8 and are indexed there (the host copy was the step's bottleneck).
    if getattr(data, 'fp_dev', None) is None or data.fp_dev.device != torch.device(device):
        data.fp_dev = torch.from_numpy(np.ascontiguousarray(np.asarray(data.fp))).to(device)
    by_mol = data.by_mol
    import threading, queue
    qu = queue.Queue(maxsize=6)

    def producer(seed2):
        # Builds every step's host-side arrays ahead of the GPU (sampling, duplicate masks, memmap gathers).
        prng = np.random.default_rng(seed2)
        for _ in range(total):
            b_ids = pd.unique(ids[prng.integers(0, len(ids), size=B * 2)])[:B]
            sr = np.array([by_mol[int(i)][prng.integers(len(by_mol[int(i)]))] for i in b_ids])
            neg = np.stack([pools[int(i)][prng.integers(0, len(pools[int(i)]), size=K)] if len(pools[int(i)]) < K
                            else prng.choice(pools[int(i)], K, replace=False) for i in b_ids])
            dup = np.isin(neg, b_ids)
            srt = np.sort(neg, 1); rep = np.zeros_like(dup); rep[:, 1:] = srt[:, 1:] == srt[:, :-1]
            if rep.any():
                for r in np.where(rep.any(1))[0]:
                    seen = set()
                    for j, k in enumerate(neg[r]):
                        if int(k) in seen: dup[r, j] = True
                        seen.add(int(k))
            order = np.argsort(sr); inv = np.empty_like(order); inv[order] = np.arange(len(sr))
            xs = np.asarray(data.x[sr[order]])[inv]; ms = np.asarray(data.meta[sr[order]])[inv]
            qu.put((b_ids, neg, dup, xs, data.lengths[sr], ms, adducts[sr]))

    threading.Thread(target=producer, args=(seed + 7919,), daemon=True).start()
    for ep in range(1, epochs + 1):
        model.train(); losses = []; start = time.monotonic()
        for step in range(spe):
            if time.monotonic() > deadline: stopped = True; break
            batch_ids, neg, dup, xs, ln, ms, ad = qu.get()
            fp = data.fp_dev[torch.as_tensor(batch_ids, device=device)].float()
            negfp = data.fp_dev[torch.as_tensor(neg.reshape(-1), device=device)].float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                z, pred = model.encode_spectra(*data.batch_np(xs, ln, ms, ad, device, augment=True))
                m = model.encode_molecules(fp); nm = model.encode_molecules(negfp).reshape(len(batch_ids), K, -1)
                positive = z @ m.T / cfg['temperature']; negative = torch.einsum('bd,bkd->bk', z, nm) / cfg['temperature']
                negative = negative.masked_fill(torch.tensor(dup, device=device), float('-inf'))
                retrieval = F.cross_entropy(torch.cat([positive, negative], 1).float(), torch.arange(len(batch_ids), device=device))
                auxiliary = F.binary_cross_entropy_with_logits(pred.float(), fp, pos_weight=pos_weight)
                loss = retrieval + cfg['aux_weight'] * auxiliary
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss.detach()))
        if not losses: break
        rec = dict(epoch=ep, loss=float(np.mean(losses)), steps=len(losses), seconds=round(time.monotonic() - start, 1),
                   lr=float(sched.get_last_lr()[0]), partial=stopped)
        ck = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch=ep, cfg=cfg, run=name)
        if cands is not None and (ep % val_every == 0 or ep == epochs or stopped):
            res, frame = validate(model, data, cands, adducts, cfg, device); rec.update(res)
            if res['mrr25'] > best:
                best = res['mrr25']; torch.save(ck, out / 'best.pt'); frame.to_csv(out / 'best_predictions.csv', index=False)
        history.append(rec); (out / 'history.json').write_text(json.dumps(history, indent=1))
        log(name, json.dumps(rec))
        if stopped: break
    torch.save(ck, out / 'last.pt')
    del model, opt
    if device.type == 'cuda': torch.cuda.empty_cache()
    return dict(best_mrr25=best, epochs=len(history), stopped=stopped, last=history[-1] if history else None)


BASE = dict(fp_bits=2048, max_peaks=96, min_peaks=3, width=192, heads=6, layers=3, embedding=192, dropout=0.1, temperature=0.1,
            aux_weight=0.5, fp_score_weight=2.0, batch_size=96, hard_negatives=16, negative_pool=128, steps_per_epoch=500,
            lr=2e-4, weight_decay=0.01, n_adducts=11, vocab='old', mol_hidden=512)
RUNS = {
    'S': dict(BASE, epochs=16, val_every=4),
    'L': dict(BASE, epochs=120, val_every=10, cosine=True, warmup=1000),
    'W': dict(BASE, epochs=120, val_every=10, cosine=True, warmup=1000, width=256, heads=8, layers=4, embedding=256,
              mol_hidden=1024, batch_size=128, n_adducts=16, vocab='new'),
}


def main():
    global FT, MD
    import subprocess
    try:
        from rdkit import rdBase
    except ImportError:
        wh = sorted(INP.rglob('rdkit-2026.3.3*.whl')) + sorted(INP.rglob('rdkit-*.whl'))
        assert wh, 'attach the RDKit 2026.03.3 wheel'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(wh[0].parent), 'rdkit==2026.3.3'], check=True)
        from rdkit import rdBase
    assert rdBase.rdkitVersion == '2026.03.3', rdBase.rdkitVersion
    SRC.mkdir(parents=True, exist_ok=True)
    (SRC / 'features.py').write_text(FEATURES_SRC); (SRC / 'model.py').write_text(MODEL_SRC)
    sys.path.insert(0, str(SRC)); import features as FT; import model as MD  # noqa: E401
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log('ENV torch', torch.__version__, device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '', 'rdkit', rdBase.rdkitVersion)
    if LOCAL:
        for r in RUNS.values(): r.update(epochs=2, val_every=1, steps_per_epoch=3, batch_size=8, hard_negatives=4, negative_pool=8, warmup=2)
    tp = sorted(INP.rglob('train.parquet')); assert len(tp) == 1, tp
    prepare(str(tp[0]), workers=os.cpu_count() or 2)
    data = Data()
    split_info = dict(train=len(data.train), val=len(data.val), np_val=int(data.mol.np_example.values[data.val].sum()),
                      np_loose=int(data.mol.loose.sum()), molecules=len(data.mol))
    log('SPLIT', split_info)
    cands = diagnostic_set(data, 60 if LOCAL else 2000); log('DIAG', len(cands))
    budget_end = T0 + float(os.environ.get('NN_HOURS', '11.3')) * 3600
    summary = dict(split=split_info, runs={})
    for name in ['S', 'L', 'W']:
        cfg = RUNS[name]
        try:
            # Leave two hours of the session for the refit.
            summary['runs'][name] = train_run(name, data, cfg, data.train, cands, cfg['epochs'], cfg['val_every'], device,
                                              deadline=_deadline(budget_end, 2.0), seed=SEED)
        except Exception as e:
            import traceback; traceback.print_exc(); summary['runs'][name] = dict(error=repr(e))
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    ok = {k: v['best_mrr25'] for k, v in summary['runs'].items() if k in ('L', 'W') and 'best_mrr25' in v}
    if ok:
        pick = max(ok, key=ok.get); cfg = dict(RUNS[pick])
        hist = json.loads((OUT / pick / 'history.json').read_text())
        epochs = cfg['epochs'] if not summary['runs'][pick]['stopped'] else len(hist)
        log('REFIT', pick, 'epochs', epochs, ok)
        try:
            summary['refit'] = dict(pick=pick, result=train_run('R_' + pick, data, cfg, np.arange(len(data.mol)), None, epochs, 10 ** 9,
                                                                device, deadline=_deadline(budget_end, 0.0), seed=SEED + 1))
        except Exception as e:
            import traceback; traceback.print_exc(); summary['refit'] = dict(pick=pick, error=repr(e))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    log('DONE', json.dumps(summary, default=str)[:3000])


def scaffold_folds(mol, k=5):
    """Every identity's Murcko group (acyclic = singleton) goes to fold sha256(seed, group) mod k."""
    group = np.where(mol.scaffold.eq(''), 'ACYCLIC:' + mol.key, mol.scaffold)
    fold_of = {g: int(hashlib.sha256((str(SEED) + 'fold' + g).encode()).hexdigest(), 16) % k for g in pd.unique(group)}
    return np.array([fold_of[g] for g in group], np.int64)


def bias_report(model, data, cands, adducts, cfg, device, unseen_mask):
    """Seen vs unseen decoys: mean within-set z-score of each decoy class (target excluded), plus MRR on the sets."""
    import torch
    model.eval(); formula = data.mol.formula.values; gaps = []; rr = []; gaps_same = []
    with torch.inference_mode():
        for i, ids in cands.items():
            sr = data.by_mol[i]; zs = 0; ps = 0; n = 0
            for st in range(0, len(sr), 256):
                b = sr[st:st + 256]
                z, p = model.encode_spectra(*data.batch(b, adducts, device)); zs = zs + z.float().sum(0); ps = ps + p.float().sum(0); n += len(b)
            fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
            s = ((model.encode_molecules(fp).float() @ (zs / n)) / cfg['temperature'] + cfg['fp_score_weight'] * (fp @ (ps / n)) / cfg['fp_bits']).cpu().numpy()
            rank = 1 + int(np.sum(s[1:] >= s[0])); rr.append(1 / rank if rank <= 25 else 0.)
            d = ids[1:]; zd = (s[1:] - s[1:].mean()) / max(s[1:].std(), 1e-6); un = unseen_mask[d]
            if un.any() and (~un).any(): gaps.append(float(zd[~un].mean() - zd[un].mean()))
            same = formula[d] == formula[i]
            if (same & un).any() and (same & ~un).any(): gaps_same.append(float(zd[same & ~un].mean() - zd[same & un].mean()))
    model.train()
    return dict(mrr25=float(np.mean(rr)), seen_minus_unseen_z=float(np.mean(gaps)) if gaps else None, n_gap=len(gaps),
                seen_minus_unseen_z_same_formula=float(np.mean(gaps_same)) if gaps_same else None, n_gap_same=len(gaps_same), n=len(rr))


def main_seeds():
    """Published recipe on every identity with new seeds (A*), and five scaffold-fold models (F*) for routing + bias."""
    global FT, MD
    import subprocess
    try:
        from rdkit import rdBase
    except ImportError:
        wh = sorted(INP.rglob('rdkit-2026.3.3*.whl')) + sorted(INP.rglob('rdkit-*.whl'))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(wh[0].parent), 'rdkit==2026.3.3'], check=True)
        from rdkit import rdBase
    assert rdBase.rdkitVersion == '2026.03.3', rdBase.rdkitVersion
    SRC.mkdir(parents=True, exist_ok=True)
    (SRC / 'features.py').write_text(FEATURES_SRC); (SRC / 'model.py').write_text(MODEL_SRC)
    sys.path.insert(0, str(SRC)); import features as FT; import model as MD  # noqa: E401
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log('ENV torch', torch.__version__, device, 'rdkit', rdBase.rdkitVersion)
    cfg = dict(BASE)
    if LOCAL: cfg.update(steps_per_epoch=3, batch_size=8, hard_negatives=4, negative_pool=8)
    epochs = 2 if LOCAL else int(os.environ.get('NN_EPOCHS', '16'))
    # Optional longer schedule for the fold models (local GPU runs): steps per epoch, warm-up + cosine.
    if os.environ.get('NN_SPE'): cfg.update(steps_per_epoch=int(os.environ['NN_SPE']))
    if os.environ.get('NN_COSINE') == '1': cfg.update(cosine=True, warmup=1000)
    if os.environ.get('NN_ARCH') == 'W':
        cfg.update(width=256, heads=8, layers=4, embedding=256, mol_hidden=1024, batch_size=128, n_adducts=16, vocab='new')
    tag = os.environ.get('NN_TAG', '')
    tp = sorted(INP.rglob('train.parquet')); assert len(tp) == 1, tp
    prepare(str(tp[0]), workers=os.cpu_count() or 2)
    data = Data(); allids = np.arange(len(data.mol)); OUT.mkdir(parents=True, exist_ok=True)
    folds = scaffold_folds(data.mol)
    pd.DataFrame(dict(key=data.mol.key, fold=folds)).to_parquet(OUT / 'fold_of_key.parquet', index=False)
    log('FOLDS', np.bincount(folds).tolist())
    budget_end = T0 + float(os.environ.get('NN_HOURS', '11.3')) * 3600
    summary = dict(folds=np.bincount(folds).tolist(), runs={})
    adducts = data.adducts(cfg['vocab'])
    for k in ([] if os.environ.get('NN_SKIP_F') == '1' else range(5)):
        name = f'F{k}{tag}'; train_ids = allids[folds != k]
        q = allids[folds == k]; rng = np.random.default_rng(SEED + k)
        npq = q[data.mol.np_example.values[q]]
        pick = np.r_[npq, rng.choice(np.setdiff1d(q, npq), min(len(q) - len(npq), (40 if LOCAL else 800) - len(npq)), replace=False)]
        pools = hard_pools(data.mol, allids, 255, only=pick); cands = {int(i): np.r_[i, pools[int(i)]] for i in pick}
        try:
            r = train_run(name, data, cfg, train_ids, None, epochs, 10 ** 9, device, deadline=_deadline(budget_end, 0.5), seed=SEED + 10 + k + int(os.environ.get('NN_SEED_OFFSET', '0')))
            ck = torch.load(OUT / name / 'last.pt', weights_only=False); m = MD.SingleModel(cfg).to(device); m.load_state_dict(ck['state_dict'])
            r['bias'] = bias_report(m, data, cands, adducts, cfg, device, unseen_mask=(folds == k))
            r['bias_np'] = bias_report(m, data, {i: v for i, v in cands.items() if data.mol.np_example.values[i]}, adducts, cfg, device,
                                       unseen_mask=(folds == k)) if len(npq) else None
            del m
        except Exception as e:
            import traceback; traceback.print_exc(); r = dict(error=repr(e))
        summary['runs'][name] = r; log(name, json.dumps(r, default=str)[:600])
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    a_lo, a_hi = (int(x) for x in os.environ.get('NN_A_RANGE', '1,5').split(','))
    for k in ([] if os.environ.get('NN_SKIP_A') == '1' else range(a_lo, a_hi)):
        name = f'A{k}'
        try:
            summary['runs'][name] = train_run(name, data, cfg, allids, None, epochs, 10 ** 9, device, deadline=_deadline(budget_end, 0.2), seed=SEED + 100 + k)
        except Exception as e:
            import traceback; traceback.print_exc(); summary['runs'][name] = dict(error=repr(e))
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    log('DONE', json.dumps(summary, default=str)[:4000])


def _deadline(budget_end, reserve_hours):
    """Monotonic deadline that leaves reserve_hours of the session budget for later stages."""
    return time.monotonic() + max(60.0, budget_end - reserve_hours * 3600 - time.time())


MODE = 'seeds'

if __name__ == '__main__':
    main_seeds() if MODE == 'seeds' else main()
