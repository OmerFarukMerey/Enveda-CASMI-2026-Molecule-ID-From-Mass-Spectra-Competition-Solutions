"""Reference model of fpgen: a transformer decoder over SMILES tokens with the conditioning fingerprint and the formula as prefix tokens, and the formula-constrained sampler.

Prefix: `n_fp` tokens from a linear map of the conditioning fingerprint (the calibrated bit probabilities of noise.py) and one token from
the element counts (9 heavy elements and hydrogens, as counts and as log1p); then BOS and the SMILES tokens, causal self-attention
throughout. The decoder tracks the heavy atoms still allowed by the target formula and masks every atom token that would exceed them
(hard constraint, `hard=True`); hydrogens cannot be counted from the string, so a sample is accepted only if its exact formula matches.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tok as T


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.h = heads; self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); self.drop = nn.Dropout(drop)

    def forward(self, x, cache=None):
        """cache: a dict with preallocated 'k', 'v' (B, heads, Lmax, dh) and the filled length 'n' (decoding, no reallocation per step)."""
        B, Tn, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, Tn, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:
            n = cache['n']; cache['k'][:, :, n:n + Tn] = k; cache['v'][:, :, n:n + Tn] = v
            k = cache['k'][:, :, :n + Tn]; v = cache['v'][:, :, :n + Tn]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None and Tn > 1)
        x = x + self.drop(self.o(y.transpose(1, 2).reshape(B, Tn, D)))
        return x + self.drop(self.ff(self.n2(x)))


class FPGen(nn.Module):
    def __init__(self, vocab: int, nbits: int = 10226, d: int = 512, layers: int = 8, heads: int = 8, drop: float = 0.1, n_fp: int = 4, max_len: int = 200):
        super().__init__()
        self.V, self.nbits, self.d, self.n_fp, self.n_pre = vocab, nbits, d, n_fp, n_fp + 1
        self.fp_in = nn.Sequential(nn.Linear(nbits, d), nn.GELU(), nn.Linear(d, n_fp * d))
        self.cnt_in = nn.Sequential(nn.Linear(2 * (T.NE + 1), d), nn.GELU(), nn.Linear(d, d))
        self.emb = nn.Embedding(vocab, d); self.pos = nn.Embedding(max_len + self.n_pre, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)
        self.drop = nn.Dropout(drop)

    def prefix(self, fp, cnt):
        f = self.fp_in(fp).view(fp.shape[0], self.n_fp, self.d)
        c = self.cnt_in(torch.cat([cnt / 10.0, torch.log1p(cnt)], -1))[:, None]
        return torch.cat([f, c], 1)

    def forward(self, fp, cnt, x):
        """fp (B, nbits), cnt (B, NE+1), x (B, L) token ids starting with BOS -> logits (B, L, V) aligned with x: row i predicts x[i + 1]."""
        h = torch.cat([self.prefix(fp, cnt), self.emb(x)], 1)
        h = self.drop(h + self.pos(torch.arange(h.shape[1], device=h.device))[None])
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))[:, self.n_pre:]

    @torch.no_grad()
    def sample(self, fp, cnt, elem, max_len=160, temperature=1.0, top_p=1.0, hard=True, greedy=False, generator=None):
        """Sample one sequence per row. elem (V, NE) heavy-element vector of every token (vocab.elem).
        -> (ids (B, L) int16, log-likelihood (B,)). With hard = True a token whose element is exhausted is masked out."""
        dev = fp.device; B = fp.shape[0]; L = max_len + self.n_pre + 1; dh = self.d // self.blocks[0].h
        h = self.prefix(fp, cnt); pos = self.pos(torch.arange(max_len + self.n_pre, device=dev))
        caches = [dict(k=torch.zeros(B, bl.h, L, dh, device=dev, dtype=h.dtype), v=torch.zeros(B, bl.h, L, dh, device=dev, dtype=h.dtype), n=0) for bl in self.blocks]
        x = h + pos[None, :self.n_pre]
        for bl, ca in zip(self.blocks, caches):
            x = bl(x, ca)
        for ca in caches:
            ca['n'] = self.n_pre
        cur = torch.full((B,), T.BOS, dtype=torch.long, device=dev)
        rem = cnt[:, :T.NE].clone()                              # heavy atoms still allowed
        out = torch.zeros(B, max_len, dtype=torch.long, device=dev); done = torch.zeros(B, dtype=torch.bool, device=dev)
        ll = torch.zeros(B, device=dev); E = elem.to(dev).float()
        for t in range(max_len):
            x = self.emb(cur)[:, None] + pos[None, self.n_pre + t:self.n_pre + t + 1]
            for bl, ca in zip(self.blocks, caches):
                x = bl(x, ca)
            for ca in caches:
                ca['n'] += 1
            z = self.head(self.norm(x))[:, 0].float()
            z[:, T.PAD] = -1e9; z[:, T.BOS] = -1e9
            if hard:
                z = z.masked_fill(((E[None] > 0) & (E[None] > rem[:, None, :])).any(-1), -1e9)
                z[:, T.EOS] = torch.where(rem.sum(-1) > 0, torch.full_like(z[:, T.EOS], -1e9), z[:, T.EOS])
            lp = torch.log_softmax(z, -1)
            if greedy:
                nxt = lp.argmax(-1)
            else:
                p = torch.softmax(z / max(temperature, 1e-3), -1)
                if top_p < 1.0:
                    s, i = torch.sort(p, -1, descending=True); c = s.cumsum(-1)
                    s = torch.where(c - s > top_p, torch.zeros_like(s), s)
                    p = torch.zeros_like(p).scatter_(1, i, s)
                p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
                nxt = torch.multinomial(p, 1, generator=generator)[:, 0]
            ll = ll + torch.where(done, torch.zeros_like(ll), lp.gather(1, nxt[:, None])[:, 0])
            nxt = torch.where(done, torch.full_like(nxt, T.PAD), nxt)
            out[:, t] = nxt; rem = (rem - E[nxt]).clamp(min=0)
            done = done | (nxt == T.EOS) | (nxt == T.PAD)
            cur = nxt
            if bool(done.all()):
                out = out[:, :t + 1]; break
        return out.to(torch.int16).cpu().numpy(), ll.cpu().numpy()


def n_params(m):
    return sum(p.numel() for p in m.parameters())
