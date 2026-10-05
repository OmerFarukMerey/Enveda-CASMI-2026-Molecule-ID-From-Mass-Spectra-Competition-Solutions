"""Local harness for the public v1 engine (casmi26-v4b-models): fold-safe FP banks, per-molecule library masks, drop_pid.

Runs the same code as nb/05_v1_engine.py + nb/06_base_lists.py on a workstation (Windows-native, no fork, no symlinks)
and adds what a leak-free study needs:
  * every fingerprint model can be swapped: the engine bank AND the three fe_v4 `model_views` models (fe_A/B/D), which
    nb/05's FP_BANK switch leaves untouched;
  * a hold-out is simulated per molecule with the engine's own interface (`exclude` mask over library spectra,
    `drop_pid`), so no purged train.parquet has to be written; `truth_mask` builds the mask by METRIC key (audit F1);
  * checkpoints are loaded with weights_only=True and pickles through a restricted unpickler (public artefacts).

    H = load_engine(v4b_dir, work_dir, bank=[fold0.pt], fe=[fold0.pt])     # fe=None -> fe_A, fe_B, fe_D
    r = run_molecule(H, spectra, target, exclude=truth_mask(H.L, key), drop_pid=-1)
    base = base_list(r, topn=60)                                           # == BASE[mid] of nb/06

Nothing here needs competition data at import time; the pure helpers are covered by test_local_engine.py.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import sys
import types
from typing import Dict, List, Optional, Sequence

import numpy as np

WORK_FILES = ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet',
              'train_fp_sel.npy', 'spec_off.npy', 'spec_mz.npy', 'spec_it.npy', 'spec_meta.parquet']
_SAFE = {('builtins', n) for n in ('dict', 'list', 'tuple', 'set', 'frozenset', 'str', 'int', 'float', 'bool', 'bytes', 'complex', 'slice',
                                   'range', 'bytearray')}
_SAFE |= {('collections', 'OrderedDict'), ('collections', 'defaultdict'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
          ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar'),
          ('numpy._core.multiarray', 'scalar'), ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer'),
          ('_codecs', 'encode')}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE or (module.startswith('numpy') and name in ('dtype', 'ndarray', 'float32', 'float64', 'int64', 'int32', 'bool_')):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')


def safe_pickle_load(f, **kw):
    """pickle.load that only rebuilds plain containers and numpy arrays (public artefacts are data, not code)."""
    return _Restricted(f, **{k: v for k, v in kw.items() if k in ('fix_imports', 'encoding', 'errors')}).load()


def safe_pickle_loads(b, **kw):
    return safe_pickle_load(io.BytesIO(b), **kw)


def _pickle_shim():
    """A stand-in for the `pickle` module whose load / loads are the restricted ones (dump etc. unchanged)."""
    shim = types.ModuleType('pickle_safe')
    shim.__dict__.update({k: v for k, v in pickle.__dict__.items() if not k.startswith('__')})
    shim.load = safe_pickle_load; shim.loads = safe_pickle_loads
    return shim


def patch_loaders(modules: Sequence[str] = ()):
    """Make the public code read its artefacts safely: torch.load -> weights_only=True; the `pickle` name inside numpy's
    npy reader and inside the given (already imported) modules -> the restricted unpickler. The global pickle module
    stays untouched (numba's on-disk cache needs it). Idempotent."""
    import torch
    shim = _pickle_shim()
    for name in ('numpy.lib.format', 'numpy.lib._format_impl') + tuple(modules):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name, fromlist=['x'])
            except ImportError:
                continue
        if hasattr(mod, 'pickle'):
            mod.pickle = shim
    if not getattr(torch.load, '_casmi_safe', False):
        _orig = torch.load

        def _load(path, *a, **k):
            k['weights_only'] = True
            return _orig(path, *a, **k)
        _load._casmi_safe = True
        torch.load = _load


def link_work(work: str, sources: Sequence[str]) -> List[str]:
    """Hard-link (same volume) or copy the engine's work files into `work`; returns the names still missing."""
    os.makedirs(work, exist_ok=True)
    missing = []
    for f in WORK_FILES:
        dst = os.path.join(work, f)
        if os.path.exists(dst):
            continue
        src = next((os.path.join(s, f) for s in sources if os.path.exists(os.path.join(s, f))), None)
        if src is None:
            missing.append(f); continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    return missing


def spectra_from_rows(rows) -> List[dict]:
    """Rows of test.parquet / train.parquet (DataFrame) -> the engine's query dicts, exactly as nb/06 builds them."""
    out = []
    for r in rows.itertuples():
        ce = r.collision_energy_ev
        ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
        out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                        mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                        ce=float(ce_arr.mean()) if len(ce_arr) else np.nan, ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
    return out


def target_mass(neutral_masses) -> Optional[float]:
    nm = np.asarray(neutral_masses, float); nm = nm[np.isfinite(nm)]
    return float(np.median(nm)) if len(nm) else None


def truth_mask(struct_key, spec_sid, spec_lib, key, lib: Optional[int] = None) -> np.ndarray:
    """Boolean mask over library spectra to hide: every spectrum whose structure has METRIC key `key` (tautomer twins
    included; class-2-like), or only those of library `lib` (class-1-like: other libraries keep their spectra)."""
    sid = np.asarray(spec_sid)
    hit = np.zeros(len(sid), bool)
    ok = sid >= 0
    hit[ok] = np.asarray(struct_key, object)[sid[ok]] == key
    if lib is not None:
        hit &= np.asarray(spec_lib) == lib
    return hit


def dedup_order(score, keys, topn: Optional[int] = None) -> List[int]:
    """Candidate indices by descending score (stable), first occurrence of each key only: nb/06's list order."""
    seen, out = set(), []
    for i in np.argsort(-np.asarray(score, float), kind='stable'):
        k = keys[i]
        if k in seen:
            continue
        seen.add(k); out.append(int(i))
        if topn and len(out) >= topn:
            break
    return out


def base_list(r: dict, topn: int = 60, score=None) -> list:
    """BASE[mid] of nb/06 from a run_molecule result: [smiles, keys, lib_max, scores, formulas, pool ids]."""
    C = r['C']; sc = r['score'] if score is None else score
    if C is None or not len(C.pid):
        return [[], [], float(r.get('lib_max', 0.0)), [], [], []]
    idx = dedup_order(sc, C.key, topn)
    return [[C.smiles[i] for i in idx], [C.key[i] for i in idx], float(r['lib_max']), [float(sc[i]) for i in idx],
            [C.formula[i] for i in idx], [int(C.pid[i]) for i in idx]]


class LogitCache:
    """Per-item cache of FPNet outputs, keyed by checkpoint and item content (peaks, precursor, adduct, CE, mode).
    The logits depend only on the query spectra, so the class-1-like, class-2-like and truth-dropped runs of one
    molecule, and the engine bank and the model_views bank when they hold the same checkpoint, share one forward pass.
    Entries live in memory and as <root>/<ckpt>/<h[:2]>/<h>.npz (float32), so several workers and later runs reuse them."""

    def __init__(self, root: str):
        self.root = root; self.mem: Dict[str, tuple] = {}; self.hits = 0; self.misses = 0

    @staticmethod
    def item_key(item: dict) -> str:
        import hashlib
        h = hashlib.sha1()
        # rounded to 1e-3: the engine bank (raw spectra) and the model_views bank (pre-cleaned spectra) build the same
        # item up to float32 rounding, so they share one entry (logits differ by float noise only)
        h.update(np.round(np.asarray(item['mz'], np.float64), 3).tobytes()); h.update(np.round(np.asarray(item['it'], np.float64), 3).tobytes())
        h.update(np.array([item['prec'], item['adduct_ix'], item['ce'], item['ce_known'], item['n_merged'], item['mode']], np.float64).tobytes())
        return h.hexdigest()

    def _path(self, tag, k):
        return os.path.join(self.root, tag, k[:2], k + '.npz')

    def get(self, tag, k):
        v = self.mem.get(tag + k)
        if v is None:
            p = self._path(tag, k)
            if os.path.exists(p):
                try:
                    with np.load(p) as z:
                        v = (z['z'], z['el'])
                    self.mem[tag + k] = v
                except Exception:
                    v = None
        return v

    def put(self, tag, k, z, el):
        self.mem[tag + k] = (z, el)
        p = self._path(tag, k); os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + f'.{os.getpid()}.tmp.npz'
        np.savez(tmp, z=z.astype(np.float32), el=el.astype(np.float32)); os.replace(tmp, p)
        if len(self.mem) > 4000:
            self.mem.clear()

    def wrap(self, fn, tag: str, each: bool = False):
        """fn(items) -> (logits (n, nbits), el (n, 10)); with each=True the bank API of fpnet6 ((1, n, nbits), (1, n, 10))."""
        def cached(items, *a, **k):
            if not items:
                return fn(items, *a, **k)
            keys = [self.item_key(it) for it in items]
            got = [self.get(tag, kk) for kk in keys]
            miss = [i for i, g in enumerate(got) if g is None]
            self.hits += len(items) - len(miss); self.misses += len(miss)
            if miss:
                z, el = fn([items[i] for i in miss], *a, **k)
                if each:
                    z, el = z[0], el[0]
                for j, i in enumerate(miss):
                    got[i] = (np.asarray(z[j], np.float32), np.asarray(el[j], np.float32)); self.put(tag, keys[i], *got[i])
            Z = np.stack([g[0] for g in got]); EL = np.stack([g[1] for g in got])
            return (Z[None], EL[None]) if each else (Z, EL)
        return cached


def _ckpt_tag(path: str) -> str:
    return f'{os.path.splitext(os.path.basename(path))[0]}_{os.path.getsize(path)}'


def load_engine(v4b: str, work: str, bank: Sequence[str], fe: Optional[Sequence[str]] = None, device: Optional[str] = None,
                amp: Optional[str] = None, generate: bool = True, verbose: bool = True, mmap: bool = True, logit_cache: Optional[str] = None, threads: int = 4):
    """Engine + fe_v4 families + public ranker, as nb/05. v4b = dataset root (code/, MANIFEST.json, fe_*.pt, ranker).
    bank: checkpoint paths of the engine's FP bank. fe: None = the dataset's fe_A / fe_B / fe_D; one path = that model
    in all three `model_views` slots (computed once). amp: 'fp16' reproduces the T4 autocast, None = device default.
    mmap: memory-map the large .npy tables (about 2.8 GB less private memory per process; numba reads them as is).
    logit_cache: directory of a LogitCache (FPNet outputs per query item, shared between runs and workers).
    threads: LightGBM threads of the public ranker (nb/05 uses 4; use 1 on a shared CPU, together with torch.set_num_threads(1))."""
    os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(os.path.dirname(os.path.abspath(work)), 'numba_cache'))
    code = os.path.join(v4b, 'code')
    for p in (os.path.join(code, 'fe_v4'), code):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import lightgbm as lgb
    import fams.derivation                      # the only public module that unpickles (prior tables, parity fixture)
    patch_loaders(('fams.derivation',))
    from casmi.library import Library
    from casmi.pool import Pool
    from casmi.engine import Engine, EngineCfg, FEATURES
    from casmi import fpnet, chem
    import v1engine
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    MAN = json.load(open(os.path.join(v4b, 'MANIFEST.json')))
    _np_load = np.load

    def _mm(path, *a, **k):                     # big .npy tables are memory-mapped: workers share them through the page cache
        if mmap and isinstance(path, str) and path.endswith('.npy') and 'mmap_mode' not in k and os.path.getsize(path) > (32 << 20):
            k['mmap_mode'] = 'r'
        return _np_load(path, *a, **k)
    np.load = _mm
    try:
        L = Library(work, verbose=verbose); P = Pool(work, verbose=verbose)
        tfp = np.load(os.path.join(work, 'train_fp_sel.npy'))
    finally:
        np.load = _np_load
    eb = fpnet.ModelBank(list(bank), device=device)
    dt = {'fp16': torch.float16, 'bf16': torch.bfloat16, None: None}[amp]
    if dt is not None:
        eb.dtype = dt
    E = Engine(L, P, tfp, EngineCfg(generate=generate), eb, verbose=verbose)
    share = fe is not None and len(fe) == 1
    fe_paths = [os.path.join(v4b, m['file']) for m in MAN['fe_models']] if fe is None else list(fe)
    V = v1engine.V1FE(E, fe_paths, MAN['families'], device=device)
    if dt is not None:
        for b in V.banks:
            b.dtype = dt
    cache = None
    if logit_cache:                             # single-checkpoint banks only (a multi-model bank averages inside)
        cache = LogitCache(logit_cache)
        if len(bank) == 1:
            eb.logits_el = cache.wrap(eb.logits_el, _ckpt_tag(bank[0]))
        for b, pth in zip(V.banks, fe_paths):
            b.logits_el_each = cache.wrap(b.logits_el_each, _ckpt_tag(pth), each=True)
    if share:                                   # one model in the A, B and D slots: run it once, repeat the logits
        _mv = V.model_views

        def model_views(prep):
            zs, es, zm, em, modes = _mv(prep)
            rep = lambda x: np.concatenate([x, x, x])
            return rep(zs), rep(es), rep(zm), rep(em), modes
        V.model_views = model_views
    with open(os.path.join(v4b, MAN['ranker']['file']), 'rb') as f:
        RK = safe_pickle_load(f)
    feats = list(RK['features']); boosters = [lgb.Booster(model_str=s) for s in RK['boosters']]

    def rank_score(X, names, threads=threads):
        pos = {n: i for i, n in enumerate(names)}
        miss = [f for f in feats if f not in pos]
        assert not miss, f'ranker features not produced: {miss}'
        Xs = np.ascontiguousarray(X[:, [pos[f] for f in feats]], dtype=np.float32)
        return np.mean([b.predict(Xs, num_threads=threads) for b in boosters], axis=0)
    return types.SimpleNamespace(E=E, V=V, L=L, P=P, MAN=MAN, FEATURES=list(FEATURES), rank_feats=feats, rank_score=rank_score, chem=chem,
                                 device=device, boosters=boosters, cache=cache)


def run_molecule(H, spectra: List[dict], target: float, exclude: Optional[np.ndarray] = None, drop_pid: int = -1) -> Dict:
    """One molecule through the engine, the families and the public ranker. Returns C (candidates), X (v1 features),
    F (family features), names, score (public ranker), lib_max, info."""
    C, X, F, names, info = H.V.run(spectra, target, exclude=exclude, drop_pid=drop_pid)
    score = np.zeros(0, np.float32)
    if len(C.pid):
        score = H.rank_score(np.concatenate([X, F], 1), list(H.FEATURES) + list(names))
    return dict(C=C, X=X, F=F, names=list(names), score=score, lib_max=float(info.get('lib_max', 0.0)), info=info)
