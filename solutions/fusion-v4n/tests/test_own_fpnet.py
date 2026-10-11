"""Our own fingerprint network and its data access (clean/fpnet.py, clean/specdata.py, clean/train_fpnet.py), without competition data.

Pinned: the parameter names and shapes of the reference checkpoint layout (so that public checkpoints load into our class and
ours load into the engine), outputs independent of padding, bit unpacking == numpy.unpackbits, the peak preparation and merge
rules, the element table, the instrument fold (timsTOF export == conditioned network), checkpoint round trip with
weights_only, and on a synthetic cache: the leak check (identity twins of held-out structures are dropped, no held-out
identity among the positives), negatives (K rows, never the positive, the window first), sampling from training spectra only,
seeded determinism of a batch, and one optimisation step of the trainer's loss on CPU.
usage: python tests/test_own_fpnet.py   (or pytest)
"""
import os, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'clean'))
import numpy as np
import pandas as pd
import torch
import fpnet, specdata

REF_LAYOUT = {'mz_emb.inv': (256,), 'nl_emb.inv': (256,), 'int_emb.weight': (512, 2), 'int_emb.bias': (512,), 'pk.weight': (512, 1536),
              'pk.bias': (512,), 'prec_emb.inv': (256,), 'ad.weight': (60, 512), 'ce_emb.inv': (64,), 'gl.weight': (512, 644), 'gl.bias': (512,),
              'blocks.0.n1.weight': (512,), 'blocks.0.qkv.weight': (1536, 512), 'blocks.0.qkv.bias': (1536,), 'blocks.0.o.weight': (512, 512),
              'blocks.0.n2.bias': (512,), 'blocks.0.ff.0.weight': (2048, 512), 'blocks.0.ff.3.weight': (512, 2048), 'norm.weight': (512,),
              'head.0.weight': (2048, 1024), 'head.3.weight': (777, 2048), 'head.3.bias': (777,), 'elem_head.0.weight': (256, 1024),
              'elem_head.2.weight': (10, 256)}


def _items(rng, n=5):
    out = []
    for _ in range(n):
        k = int(rng.integers(1, 300)); prec = float(rng.uniform(150, 900))
        out.append(fpnet.make_item(np.sort(rng.uniform(40, prec + 10, k)), rng.uniform(0.0005, 1, k), prec, int(rng.integers(0, 60)),
                                   float(rng.uniform(5, 80)) if rng.random() < 0.6 else np.nan, int(rng.integers(1, 12)), 1 if rng.random() < 0.5 else -1))
    return out


def test_reference_layout():
    sd = fpnet.FPNet(777, layers=1).state_dict()
    assert len(sd) == 11 + 12 + 2 + 4 + 4, len(sd)
    for k, s in REF_LAYOUT.items():
        assert tuple(sd[k].shape) == s, (k, tuple(sd[k].shape))
    m = fpnet.FPNet(10226)
    assert sum(p.numel() for p in m.parameters()) == 49_686_780, sum(p.numel() for p in m.parameters())
    assert abs(float(m.mz_emb.inv[0]) - 628.31854) < 1e-3 and abs(float(m.mz_emb.inv[-1]) - 0.00314905) < 1e-7
    assert abs(float(m.ce_emb.inv[0]) - 6.2831855) < 1e-5 and abs(float(m.ce_emb.inv[-1]) - 0.03964422) < 1e-7


def test_padding_invariance_and_items():
    rng = np.random.default_rng(0); torch.manual_seed(0)
    m = fpnet.FPNet(64, d=64, layers=2, heads=4).eval(); items = _items(rng, 6)
    a = fpnet.collate(items); b = fpnet.collate(items, pad_to=fpnet.MAX_PEAKS)
    assert a[0].shape[1] == max(len(x['mz']) for x in items) <= fpnet.MAX_PEAKS and b[0].shape[1] == 160
    with torch.no_grad():
        za, ea = m(*a); zb, eb = m(*b); one = m(*fpnet.collate(items[2:3]))[0]
    assert za.shape == (6, 64) and ea.shape == (6, 10)
    assert float((za - zb).abs().max()) < 1e-4 and float((ea - eb).abs().max()) < 1e-4 and float((one - za[2:3]).abs().max()) < 1e-4
    it = fpnet.make_item([100.0, 200.0], [1.0, 0.25], 300.0, 3, np.nan, 20, -1)
    assert it['ce'] == 0.0 and it['ce_known'] == 0.0 and it['n_merged'] == 8 and it['mode'] == -1.0 and np.allclose(it['it'], [1.0, 0.5])
    assert float(fpnet.collate([it])[7][0]) == 2.0


def test_unpack_bits():
    rng = np.random.default_rng(1); bits = rng.random((7, 3, 77)) < 0.3
    got = fpnet.unpack_bits(torch.from_numpy(np.packbits(bits, axis=-1)), 77).numpy()
    assert got.shape == (7, 3, 77) and np.array_equal(got.astype(bool), bits)


def test_prep_peaks():
    mz = np.array([0.0, 50.0, 60.0, 70.0, 302.0, 302.5]); it = np.array([9.0, 1.0, 4.0, 0.003, 0.5, 100.0])
    pm, pi = fpnet.prep_peaks(mz, it, 300.0)                         # m/z 0 and > precursor + 2 dropped, floor 0.001 x 4.0
    assert np.allclose(pm, [50, 60, 302]) and np.allclose(pi, np.sqrt(np.array([1.0, 4.0, 0.5]) / 4.0))
    rng = np.random.default_rng(2); mz = np.sort(rng.uniform(50, 500, 400)); it = rng.uniform(0.01, 1, 400)
    pm, pi = fpnet.prep_peaks(mz, it, 600.0)
    assert len(pm) == 160 and np.all(np.diff(pm) > 0) and np.isclose(pi.max(), 1.0) and np.isclose(pi.min() ** 2 * it.max(), np.sort(it)[-160])
    assert len(fpnet.prep_peaks([], [], 100.0)[0]) == 0


def test_merge_and_elements():
    a = (np.array([100.0, 200.0, 300.0]), np.array([1.0, 0.5, 0.2])); b = (np.array([100.006, 200.002, 400.0]), np.array([0.2, 1.0, 0.15]))
    mz, it = specdata.merge_spectra([a, b])                          # 0.005 Da: the more intense peak's m/z, the larger intensity
    assert np.allclose(mz, [100.0, 100.006, 200.002, 300.0, 400.0], atol=1e-4) and np.allclose(it, [1.0, 0.2, 1.0, 0.2, 0.15])
    mz, it = specdata.merge_spectra([(np.array([900.0]), np.array([1.0])), (np.array([900.02]), np.array([0.5]))])
    assert np.allclose(it, [1.0, 1.0])                               # every spectrum is scaled to its base peak first
    assert len(specdata.merge_spectra([a, b], max_peaks=2)[0]) == 2
    e = specdata.element_targets(['C17H20ClN5O', 'CH4', '', 'C2Br2Na'])
    assert np.allclose(e[0], np.log1p([17, 20, 5, 1, 0, 0, 0, 1, 0, 0])) and np.allclose(e[3], np.log1p([2, 0, 0, 0, 0, 0, 0, 0, 2, 0])) and e[2].sum() == 0
    assert [specdata.instrument_class(x) for x in ('timsTOF Pro', 'Orbitrap', 'ESI-QTOF', 'Ion Trap', 'QqQ', 'unknown')] == [0, 1, 2, 3, 4, 5]


def test_instrument_fold_and_checkpoint():
    rng = np.random.default_rng(3); torch.manual_seed(1)
    m = fpnet.FPNet(32, d=64, layers=1, heads=4, n_instr=6).eval(); torch.nn.init.normal_(m.ins.weight)
    inp = fpnet.collate(_items(rng, 4))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        p = os.path.join(t, 'a.pt'); fpnet.save_ckpt(p, m, 7, dict(args=dict(lr=1e-3), licence='ours'))
        m2 = fpnet.load_ckpt(p); ck = torch.load(p, weights_only=True)
        assert ck['step'] == 7 and ck['nbits'] == 32 and ck['heads'] == 4 and m2.n_instr == 6
        with torch.no_grad():
            z = m(*inp, instr=torch.zeros(4, dtype=torch.long))[0]
            assert float((m2(*inp, instr=torch.zeros(4, dtype=torch.long))[0] - z).abs().max()) == 0.0
            plain = fpnet.FPNet(32, d=64, layers=1, heads=4).eval(); plain.load_state_dict(fpnet.fold_instrument(m.state_dict(), 0))
            assert float((plain(*inp)[0] - z).abs().max()) < 1e-5
            assert float((plain(*inp)[0] - m(*inp, instr=torch.full((4,), 2))[0]).abs().max()) > 1e-3


def _toy(t, n_struct=40, twin=True):
    """Synthetic cache: structure s has key K<s>, mass 100 + s // 4 (four isomers per window); structures 0..7 are held out; with
    `twin`, training structure 39 has the metric key of held-out structure 3 (a tautomer twin)."""
    rng = np.random.default_rng(5); keys = [f'K{s:013d}' for s in range(n_struct)]; raw = list(keys)
    if twin:
        keys[39] = keys[3]
    st = pd.DataFrame(dict(sid=np.arange(n_struct), inchikey14=raw, key=keys, rawkey=raw)); st.to_parquet(os.path.join(t, 'train_structs.parquet'))
    pk = sorted(set(keys)) + [f'C{c:013d}' for c in range(60)]            # catalogue-only rows
    mass = np.array([100.0 + int(k[1:]) // 4 if k[0] == 'K' else 100.0 + int(k[1:]) // 6 + 1e-5 for k in pk])
    o = np.argsort(mass, kind='stable'); pk = [pk[i] for i in o]; mass = mass[o]
    pd.DataFrame(dict(key=pk, formula=['C6H12O6'] * len(pk), mass=mass, src=[1 if k[0] == 'K' else 2 for k in pk],
                      train_sid=[int(k[1:]) if k[0] == 'K' else -1 for k in pk])).to_parquet(os.path.join(t, 'pool_meta.parquet'))
    np.save(os.path.join(t, 'pool_fp.npy'), np.packbits(rng.random((len(pk), 40)) < 0.4, axis=1)); np.save(os.path.join(t, 'fp_bits.npy'), np.arange(40, dtype=np.int32))
    pd.DataFrame(dict(sid=np.arange(8), fold=['hold'] * 6 + ['np'] * 2)).to_parquet(os.path.join(t, 'split.parquet'))
    sid = np.repeat(np.arange(n_struct), 5); n = len(sid); ln = rng.integers(0, 30, n); ln[7] = 0
    np.save(os.path.join(t, 'spec_off.npy'), np.r_[0, np.cumsum(ln)].astype(np.int64))
    np.save(os.path.join(t, 'spec_mz.npy'), np.concatenate([np.sort(rng.uniform(40, 99, k)) for k in ln]).astype(np.float32))
    np.save(os.path.join(t, 'spec_it.npy'), rng.uniform(0.01, 1, int(ln.sum())).astype(np.float32))
    pd.DataFrame(dict(precursor_mz=101.0 + sid // 4, ce_mean=np.where(rng.random(n) < 0.5, 30.0, np.nan), ce_n=rng.integers(0, 3, n), sid=sid,
                      lib=rng.integers(0, 11, n), instr=rng.integers(0, 6, n), adduct_ix=rng.integers(0, 10, n), mode=np.where(rng.random(n) < 0.7, 1, -1),
                      n_clean=ln)).to_parquet(os.path.join(t, 'spec_meta.parquet'))


def test_specdata_leak_check_sampling_and_step():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        _toy(t); msgs = []
        D = specdata.SpecData(t, K=7, log=msgs.append)
        assert 'LEAK CHECK' in msgs[0] and D.leak == 0 and D.leak_pid == 0 and D.n_twin == 1
        tr_sids = set(D.sid[D.tr].tolist())
        assert tr_sids == set(range(8, 39)) and (D.n_clean[D.tr] > 0).all()        # held-out 0..7 and the twin 39 never train
        assert D.sid2pid[39] == D.sid2pid[3] and (D.fold[:8] != 'train').all()
        rng = np.random.default_rng(0)
        for _ in range(20):
            ids = D.sample_ids(rng, 8); assert D.in_tr[ids].all() and len(ids) == 8
        pos = D.sid2pid[D.sid[D.tr[:16]]]; neg = D.negatives(pos, rng)
        assert neg.shape == (16, 7) and not (neg == pos[:, None]).any() and all(len(set(r)) == 7 for r in neg.tolist())
        for p, r in zip(pos, neg):                                                # the window's other rows come first
            lo, hi = D.window(p); inside = np.sum((r >= lo) & (r < hi)); assert inside == min(7, hi - lo - 1)
        b1 = D.batch(np.random.default_rng(9), 8); b2 = D.batch(np.random.default_rng(9), 8)
        assert all(torch.equal(x, y) for x, y in zip(b1[0], b2[0])) and np.array_equal(b1[2], b2[2]) and np.array_equal(b1[3], b2[3])
        D.restrict(specdata.subset(D, 'tims')); assert (D.instr[D.sample_ids(rng, 8)] == 0).all()
        s = specdata.BatchStream(t, None, ('hold', 'np'), 0, 0, 8, 0.3, specdata.AUG_STD, 'instr_bal', K=7)
        arrs, pos, neg, ids = next(s.stream(0))
        # one step of the trainer's loss on CPU
        torch.manual_seed(0); m = fpnet.FPNet(D.nbits, d=32, layers=1, heads=4); opt = torch.optim.AdamW(m.parameters(), 1e-3)
        PFP = torch.from_numpy(np.load(os.path.join(t, 'pool_fp.npy'))); cand = torch.as_tensor(np.concatenate([pos[:, None], neg], 1))
        z, el = m(*arrs); fp = fpnet.unpack_bits(PFP[cand], D.nbits); sc = torch.einsum('bkn,bn->bk', fp, z)
        loss = torch.nn.functional.cross_entropy(sc, torch.zeros(8, dtype=torch.long)) + torch.nn.functional.binary_cross_entropy_with_logits(z, fp[:, 0]) \
            + 0.1 * torch.nn.functional.smooth_l1_loss(el, torch.from_numpy(D.elem)[cand[:, 0]])
        loss.backward(); opt.step(); assert np.isfinite(loss.item())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:                                      # a split that misses an identity twin still cannot leak
        _toy(t, twin=True); D = specdata.SpecData(t, K=7, holdout=('hold',), log=lambda *a: None)
        assert 3 not in set(D.sid[D.tr].tolist()) and 39 not in set(D.sid[D.tr].tolist()) and 6 in set(D.sid[D.tr].tolist())


def test_trainer_helpers():
    import train_fpnet
    a = train_fpnet.get_args(['--work', 'w', '--out', 'o'])
    assert a.max_steps == 82855 and a.bs == 256 and a.duty == 0.8 and a.game_duty == 0.25 and a.yield_gb == 9.0 and a.max_gb == 8.0 and a.holdout == 'hold,np'

    class D: sid = np.array([1, 1, 1, 2, 2, 3] + [4] * 9); mode = np.array([1, 1, -1, 1, 1, 1] + [1] * 9)
    g = train_fpnet.groups_of(D, np.ones(15, bool), 10, np.random.default_rng(0))
    assert g == [[2], [0, 1], [3, 4], [5], list(range(6, 14))]


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
    print('own fpnet OK: reference layout, padding invariance, bit unpacking, peak rules, merge, elements, instrument fold, checkpoint, '
          'leak check and samplers on a synthetic cache, one loss step')
