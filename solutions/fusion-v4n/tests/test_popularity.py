"""fusion_core.popularity_reorder must match bobthebot369's v10 popularity prior (LB 0.401) exactly.

usage: python tests/test_popularity.py   (or pytest)
"""
import os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nb', 'embed'))
import numpy as np
import fusion_core


def bob(smis, keys, scs, forms, pids_, POOL_POP, POOLPOP_MU):
    # verbatim logic of the v10 notebook cell (Apache-2.0), wrapped in a function
    s_ = np.asarray(scs, np.float64); z_ = (s_ - s_.mean()) / (s_.std() + 1e-9)
    pid_ = np.asarray(pids_); isg_ = pid_ < 0
    f_ = z_ + POOLPOP_MU * np.where(isg_, 0.0, POOL_POP[np.maximum(pid_, 0)])
    pool_i = np.where(~isg_)[0]; o_ = np.empty(len(pid_), np.int64)
    o_[isg_] = np.where(isg_)[0]; o_[~isg_] = pool_i[np.argsort(-f_[pool_i], kind='stable')]
    smis = [smis[i] for i in o_]; keys = [keys[i] for i in o_]; scs = [float(f_[i]) for i in o_]
    forms = [forms[i] + ('|gen' if isg_[i] else '') for i in o_]
    return smis, keys, scs, forms


def test_matches_bob():
    rng = random.Random(0)
    pool_pop = np.array([rng.uniform(0, 8) for _ in range(1000)])
    for _ in range(200):
        n = rng.randint(2, 60)
        smis = [f's{i}' for i in range(n)]; keys = [f'k{i}' for i in range(n)]
        scs = sorted((rng.gauss(0, 1) for _ in range(n)), reverse=True)
        forms = [rng.choice(['C10H12O2', 'C9H8O4', 'C15H10O5']) for _ in range(n)]
        pids = [rng.randint(0, 999) if rng.random() > 0.2 else -1 for _ in range(n)]
        for mu in (0.1, 0.15, 0.25):
            a = fusion_core.popularity_reorder(smis, keys, scs, forms, pids, pool_pop, mu)[:4]
            b = bob(smis, keys, scs, forms, pids, pool_pop, mu)
            assert a == b
    print('popularity_reorder == bobthebot v10 on 200 random lists x 3 weights')


if __name__ == '__main__':
    test_matches_bob()
