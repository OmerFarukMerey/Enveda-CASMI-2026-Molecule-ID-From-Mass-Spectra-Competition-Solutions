"""Pure-torch stand-in for the two pygmtools functions GLACIER's training losses call (own code, no pygmtools source copied).

pygmtools (Wang et al., MulanPSL-2.0) is not installed and nothing may be installed without approval. ms-pred's JointModel uses:
  - sinkhorn(s, n1, n2, tau, backend='pytorch')  in the contrastive loss (rank-1 probability of the true structure among decoys);
  - hungarian(s, n2=..., backend='pytorch')       in the fragment loss (not used by this fine-tune: frag_weight 0, see README).
Semantics implemented (pygmtools defaults: max_iter 10, dummy_row False, non-batched operation): per batch element b the
[:n1[b], :n2[b]] block of s / tau is normalised in log space, alternating rows (even iterations) and columns (odd iterations);
entries outside the block are 0 in the output. hungarian returns the 0/1 matrix of scipy's maximum-weight assignment per block.
"""
import numpy as np
import torch

__glacierft_shim__ = True
__version__ = '0.0-glacierft-shim'


def _sizes(n, default, B, device):
    if n is None:
        return [int(default)] * B
    if torch.is_tensor(n):
        return [int(x) for x in n.reshape(-1).tolist()]
    return [int(x) for x in np.asarray(n).reshape(-1)]


def sinkhorn(s, n1=None, n2=None, unmatch1=None, unmatch2=None, dummy_row=False, max_iter=10, tau=1.0,
             batched_operation=False, backend='pytorch'):
    if unmatch1 is not None or unmatch2 is not None or dummy_row:
        raise NotImplementedError('pygmtools shim: unmatch / dummy_row not implemented')
    squeeze = s.dim() == 2
    if squeeze:
        s = s.unsqueeze(0)
    B = s.shape[0]
    r, c = _sizes(n1, s.shape[1], B, s.device), _sizes(n2, s.shape[2], B, s.device)
    log_s = s / tau
    out = torch.full_like(log_s, -float('inf'))
    for b in range(B):
        x = log_s[b, :r[b], :c[b]]
        for i in range(int(max_iter)):
            if i % 2 == 0:
                x = x - torch.logsumexp(x, 1, keepdim=True)
            else:
                x = x - torch.logsumexp(x, 0, keepdim=True)
        out[b, :r[b], :c[b]] = x
    out = torch.exp(out)
    return out[0] if squeeze else out


def hungarian(s, n1=None, n2=None, unmatch1=None, unmatch2=None, nproc=1, backend='pytorch'):
    from scipy.optimize import linear_sum_assignment
    squeeze = s.dim() == 2
    if squeeze:
        s = s.unsqueeze(0)
    B = s.shape[0]
    r, c = _sizes(n1, s.shape[1], B, s.device), _sizes(n2, s.shape[2], B, s.device)
    a = s.detach().float().cpu().numpy()
    perm = np.zeros(a.shape, dtype=np.float32)
    for b in range(B):
        blk = a[b, :r[b], :c[b]]
        if blk.size:
            ri, ci = linear_sum_assignment(blk, maximize=True)
            perm[b, ri, ci] = 1.0
    out = torch.from_numpy(perm).to(device=s.device, dtype=s.dtype)
    return out[0] if squeeze else out
