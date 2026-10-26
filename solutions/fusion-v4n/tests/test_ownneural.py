"""Data-free tests of the ownneural model (clean/onnet.py, clean/onnet_infer.py) and its tools (train/ownneural/: split roles, negative
tables, list metric, evaluation helpers).

Pinned down: spectrum arrays keep the most intense peaks in m/z order and map polarity to 1/0; the listwise + in-batch loss is finite,
ignores masked negatives and falls when the true structure is closer; a CPU training step of the network runs and lowers the loss on a
repeated batch; a molecule's embedding does not depend on the order of its spectra nor on the other molecules of the batch; checkpoints
round-trip; the raw spectrum adapter cleans peaks and maps adduct / polarity / collision energy / instrument; candidate bits select the
pool fingerprint and flag bad SMILES; split roles (identity, scaffold, validation groups, H); Tanimoto ranking; MRR with ties; the
top-1-locked re-rank never moves rank 1.
usage: python tests/test_ownneural.py   (or pytest)
"""
import os, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'clean')); sys.path.insert(0, os.path.join(SOL, 'train', 'ownneural'))
import numpy as np
import pandas as pd
import torch

import onnet
import onnet_infer as OI


def _batch(B=4, K=3, nbits=64, seed=0):
    rng = np.random.default_rng(seed); peaks, prec, add, mode, ce, instr, own = [], [], [], [], [], [], []
    for b in range(B):
        for _ in range(1 + b % 3):
            n = int(rng.integers(3, 9)); mz = np.sort(rng.uniform(50, 300, n)).astype(np.float32); it = rng.uniform(0.01, 1, n).astype(np.float32)
            peaks.append((mz, it)); prec.append(320.0 + b); add.append(b % 3); mode.append(1 if b % 2 else -1); ce.append(np.nan if b == 0 else 30.0); instr.append(0); own.append(b)
    arr = onnet.spectra_arrays(peaks, prec, add, mode, ce, instr, own, max_peaks=6)
    fp = lambda *s: np.packbits(rng.integers(0, 2, s + (nbits,)).astype(np.uint8), axis=-1)
    arr.update(pos_fp=fp(B), neg_fp=fp(B, K), neg_ok=np.ones((B, K), bool), pid=np.arange(B), w=np.ones(B, np.float32))
    arr['neg_ok'][0, -1] = False
    return onnet.to_torch(arr, 'cpu')


def _net(nbits=64, **kw):
    torch.manual_seed(0)
    return onnet.ONNet(nbits=nbits, d=32, layers=1, out=16, hidden=(32,), dropout=0.0, **kw)


def test_spectra_arrays():
    pk = [(np.array([10., 20., 30., 40.]), np.array([.1, 1., .5, .2])), (np.array([5.]), np.array([1.]))]
    b = onnet.spectra_arrays(pk, [100., 50.], [0, 1], [1, -1], [np.nan, 20.], [0, 1], [0, 0], max_peaks=2)
    assert b['mz'][0].tolist() == [20., 30.] and b['it'][0].tolist() == [1., .5] and b['n_peaks'].tolist() == [2, 1]
    assert b['mode'].tolist() == [1, 0] and b['ce_known'].tolist() == [0., 1.] and b['ce'].tolist() == [0., 20.]


def test_loss_masks_and_direction():
    zs = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1); zp = zs.clone(); zn = torch.nn.functional.normalize(torch.randn(3, 2, 8), dim=-1)
    ok = torch.ones(3, 2, dtype=torch.bool); w = torch.ones(3); pid = torch.arange(3)
    l_good, acc = onnet.list_nce_loss(zs, zp, zn, pid, ok, w, torch.tensor(20.0))
    l_bad, _ = onnet.list_nce_loss(zs, -zp, zn, pid, ok, w, torch.tensor(20.0))
    assert torch.isfinite(l_good) and l_good < l_bad and float(acc) == 1.0
    zn2 = zn.clone(); zn2[:, 1] = zs                                   # a masked negative identical to the truth must not count
    ok2 = ok.clone(); ok2[:, 1] = False
    l_m, _ = onnet.list_nce_loss(zs, zp, zn2, pid, ok2, w, torch.tensor(20.0), w_nce=0.0)
    l_r, _ = onnet.list_nce_loss(zs, zp, zn[:, :1], pid, ok[:, :1], w, torch.tensor(20.0), w_nce=0.0)
    assert torch.allclose(l_m, l_r, atol=1e-6)


def test_train_step_cpu():
    m = _net(w_aux=0.5); b = _batch(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3); losses = []
    for _ in range(25):
        loss, _ = m(b); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    assert np.isfinite(losses).all() and losses[-1] < losses[0]


def test_embedding_invariances():
    m = _net().eval(); b = _batch(); B = 4
    z, _ = onnet.embed_molecules(m, {k: v.numpy() for k, v in b.items() if k in ('mz', 'it', 'n_peaks', 'prec', 'add', 'mode', 'ce', 'ce_known', 'instr', 'owner')}, B, 'cpu')
    arr = {k: v.numpy() for k, v in b.items() if k in ('mz', 'it', 'n_peaks', 'prec', 'add', 'mode', 'ce', 'ce_known', 'instr', 'owner')}
    perm = np.random.default_rng(1).permutation(len(arr['owner']))
    z2, _ = onnet.embed_molecules(m, {k: v[perm] for k, v in arr.items()}, B, 'cpu', chunk=3)
    assert torch.allclose(z, z2, atol=1e-5)
    keep = arr['owner'] == 2; sub = {k: v[keep] for k, v in arr.items()}; sub['owner'] = np.zeros(keep.sum(), np.int64)
    z3, _ = onnet.embed_molecules(m, sub, 1, 'cpu')
    assert torch.allclose(z[2], z3[0], atol=1e-5)


def test_ckpt_roundtrip(tmp_path=None):
    import tempfile
    m = _net().eval(); d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path); p = os.path.join(d, 'x.pt'); onnet.save_ckpt(p, m, dict(step=3))
    m2 = onnet.load_ckpt(p); fp = np.packbits(np.eye(64, dtype=np.uint8)[:5], axis=1)
    assert m2.meta['step'] == 3 and torch.allclose(onnet.embed_structs(m, fp, 'cpu'), onnet.embed_structs(m2, fp, 'cpu'), atol=2e-3)


def test_raw_adapter():
    f = OI.spectrum_fields([50., 100., 200., 400.], [10., 1000., 0.5, 50.], 300.0, '[M+CH2O2-H]-', 'negative', np.array([20., np.nan, 40.]), 'Bruker timsTOF Pro')
    assert f[0].tolist() == [50., 100.] and abs(f[1][1] - 1.0) < 1e-7 and f[3] == 8 and f[4] == 0 and f[5] == 30.0 and f[6] == 0
    assert OI.spectrum_fields([50.], [1.], 300.0, '[M+Foo]+', 'positive', None, 'Orbitrap')[3:] == (59, 1, OI.spectrum_fields([50.], [1.], 300.0, '[M+Foo]+', 'positive', None, 'Orbitrap')[5], 1)
    df = pd.DataFrame(dict(ms2_mzs=[np.array([50., 100.]), np.array([900.])], ms2_normalized_intensities=[np.array([.5, 1.]), np.array([1.])], precursor_mz=[300., 300.],
                           adduct=['[M+H]+', '[M+H]+'], ionization_mode=['positive', 'positive'], collision_energy_ev=[np.array([20.]), np.array([20.])], instrument_type=['timsTOF', 'timsTOF']))
    a = OI.arrays_from_rows([df, df.iloc[1:]])
    assert a['owner'].tolist() == [0, 1] and a['n_peaks'].tolist() == [2, 0]     # molecule 0 drops its empty spectrum; molecule 1 keeps its only one (metadata token)


def test_cand_bits():
    import pool_build
    bits = np.array([0, 5, 4096 + 7, pool_build.FP_RAW - 1]); cb = OI.CandBits(bits)
    fp, ok = cb(['CCO', 'not a smiles', 'CCO'])
    assert ok.tolist() == [True, False, True] and (fp[0] == fp[2]).all() and fp.shape == (3, 1) and not fp[1].any()
    raw = np.unpackbits(pool_build.record(pool_build.standardise('CCO')[0])[1])[:pool_build.FP_RAW]
    assert np.unpackbits(fp[0])[:4].tolist() == raw[bits].tolist()


def test_split_roles():
    import build_split as BS
    st = pd.DataFrame(dict(sid=range(8), inchikey14=list('ABCDEFGH'), key=list('ABCDEFGH'), rawkey=list('ABCDEFGH')))
    grp = np.array(['s1', 's1', 's2', 's3', 's3', 'k:F', 's4', 's4'])
    env = np.array([1, 1, 1, 1, 0, 1, 1, 1], bool)
    role = BS.assign(st, grp, {'A'}, {'G'}, env, n_val=10, max_group=5, seed=0)
    assert role[0] == 'eval' and role[1] == 'eval_scaf' and role[6] == 'H' and role[7] == 'H_scaf'
    assert role[3] == 'train' and role[4] == 'train'            # group s3 has a member without Enveda spectra: never validation
    assert role[2] == 'val' and role[5] == 'val'
    role = BS.assign(st, grp, {'A'}, {'G'}, env, n_val=10, max_group=5, seed=0, e_sids={3, 0})
    assert role[3] == 'H_scaf' and role[0] == 'eval' and role[4] == 'train'   # an E structure leaves training; eval keeps its role


def test_tanimoto_ranking():
    import build_negs as BN
    X = np.packbits(np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0]], np.uint8), axis=1)
    U = BN.as_u64(X); t = BN.tanimoto(U[0], U)
    assert np.allclose(t, [1.0, 0.75, 0.0, 0.5])
    assert BN.top_similar(U[0], np.array([10, 11, 12]), U[1:], 2).tolist() == [10, 12]
    assert BN.top_similar(U[0], np.array([], np.int64), U[:0], 3).tolist() == [-1, -1, -1]


def test_list_metrics():
    import ondata as OD
    import eval_lists as EL
    mrr, rr = OD.list_mrr(np.array([0.5, 0.9, 0.1, 0.5, 0.2]), np.array([0, 0, 0, 1, 1]))
    assert np.allclose(rr, [0.5, 1.0]) and abs(mrr - 0.75) < 1e-12
    assert abs(EL.rr_among([0.2, 0.5, 0.5, np.nan], 1) - 0.5) < 1e-12     # NaN -> median 0.5: two ties
    assert EL.ranks_desc([3.0, np.nan, 1.0]).tolist() == [1.0, 2.0, 3.0]
    v = np.array([0.0, 0.1, 0.9, 0.5, 0.6, 0.7, 0.4, 0.3])  # 7 movable rows: base rank 2 scored last, base rank 3 first -> they swap
    assert EL.locked_pos(v, 3) == 2 and EL.locked_pos(v, 1) == 1 and EL.locked_pos(v, 2) == 3
    v = np.array([0.0, 0.1, 0.2, 0.9])                     # 3 movable rows: rank 2 cannot be displaced (1/3 + 0.25/4 > 1/4 + 0.25/2)
    assert EL.locked_pos(v, 3) == 3
    assert EL.locked_pos(np.array([0.0, 0.9, 0.1, 0.2, 0.95]), 5) == 4    # fused 0.417 / 0.300 / 0.263 / 0.292: rank 5 passes rank 4 only


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
    print('ALL OK')
