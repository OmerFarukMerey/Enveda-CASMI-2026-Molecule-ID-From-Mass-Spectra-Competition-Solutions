"""POP_SPLIT: the popularity prior without the substance records that spectral libraries deposited themselves (reviews/popsplit-2026-10-08.md).

Today's prior per pool row is log1p(PubChem substance records) + log1p(PubMed articles). A compound that is in a public spectral library
gets one or more of its substance records from that library's own deposit (MassBank Europe, MoNA, NIST Mass Spectrometry Data Center,
RIKEN NPDepo are PubChem depositors), so for little-documented compounds the count partly says "is in a library". With the flag on, the
substance part of the rows listed in the dataset casmi26-pop-split becomes log1p(max(sid - spec, 0)); every other row keeps its value
bit for bit, and the PubMed part is never touched. Only counts of public PubChem records are read: no spectrum, no library content.

    pool_pop(lsid, lpmid, None)        -> lsid + lpmid                                     (flag off: today's array, exactly)
    pool_pop(lsid, lpmid, table_dir)   -> the same with the listed rows replaced; (array, number of rows changed)
table_dir holds pool_split.parquet (pid, sid, spec: pool rows with spec > 0) and meta.json (n_pool: the pool length the pids refer to).
"""
import json, os

import numpy as np


def load_table(table_dir):
    import pandas as pd
    t = pd.read_parquet(os.path.join(table_dir, 'pool_split.parquet'), columns=['pid', 'sid', 'spec'])
    meta = json.load(open(os.path.join(table_dir, 'meta.json'), encoding='utf-8'))
    return t.pid.values.astype(np.int64), t.sid.values.astype(np.int64), t.spec.values.astype(np.int64), int(meta['n_pool'])


def split_values(lsid, lpmid, pid, sid, spec, n_pool):
    """lsid + lpmid with rows pid set to log1p(max(sid - spec, 0)) + lpmid[pid]. Raises when the table does not fit the pool."""
    lsid = np.asarray(lsid, np.float64); lpmid = np.asarray(lpmid, np.float64)
    if len(lsid) != n_pool or len(lpmid) != n_pool:
        raise ValueError(f'pop_split: the table was built for a pool of {n_pool} rows, the arrays have {len(lsid)} / {len(lpmid)}')
    if len(pid) and (pid.min() < 0 or pid.max() >= n_pool or len(np.unique(pid)) != len(pid)):
        raise ValueError('pop_split: pool ids out of range or repeated')
    if (spec < 0).any() or (sid < 0).any():
        raise ValueError('pop_split: negative counts')
    out = lsid + lpmid
    out[pid] = np.log1p(np.maximum(sid - spec, 0).astype(np.float64)) + lpmid[pid]
    return out, int(len(pid))


def pool_pop(lsid, lpmid, table_dir=None):
    """The array handed to fusion_core.build_submission(pool_pop=...). table_dir None: (lsid + lpmid, 0), today's prior."""
    if table_dir is None:
        return np.asarray(lsid, np.float64) + np.asarray(lpmid, np.float64), 0
    return split_values(lsid, lpmid, *load_table(table_dir))
