"""Measurement suite (idea.md section 8). Everything that only needs saved test outputs works on CPU numpy; the model-based
probes (head importance 8.4, gradient share 8.5, swap tests 8.3) take a trained model and a CropDataset."""
import itertools
from math import comb

import numpy as np
import torch
import torch.nn.functional as F

from .data import prep


# ----------------------------------------------------------------------------- 8.2 routing structure
def entropy(p, axis=-1):
    return -(p * np.log(p + 1e-12)).sum(axis)


def routing_specialization(pi):
    """S^l = H(mean pi) - mean H(pi) = I(sample; head).  pi [n,Lr,H] -> S [Lr]."""
    return entropy(pi.mean(0)) - entropy(pi).mean(0)


def specialization_null(model, z, device, tau=1.0, seed=0):
    """Decorrelated-z reference for H2: permute every z feature independently across samples (keeps each feature's
    marginal, destroys the clip-specific structure), push through the trained routers, recompute S.
    NOTE: permuting whole z rows would leave S unchanged (S depends only on the multiset of pi)."""
    rng = np.random.default_rng(seed)
    zp = np.stack([rng.permutation(z[:, j]) for j in range(z.shape[1])], 1).astype(np.float32)
    zt = torch.from_numpy(zp).to(device)
    pis = []
    with torch.no_grad():
        for r in range(model.n_routed):
            _, pi, _, _ = model.routers[r].gate(zt, tau, True)
            pis.append(pi.cpu().numpy())
    return routing_specialization(np.stack(pis, 1))


def _codes(a):
    return np.unique(a, return_inverse=True)[1].reshape(-1)


def _H_mm(codes, n):
    c = np.bincount(codes)
    c = c[c > 0]
    p = c / n
    return -(p * np.log(p)).sum() + (len(c) - 1) / (2 * n)   # plug-in + Miller-Madow


def mi_mm(a, b):
    a, b = _codes(a), _codes(b)
    n = len(a)
    return _H_mm(a, n) + _H_mm(b, n) - _H_mm(a * (b.max() + 1) + b, n)


def cond_mi_mm(r, c, y):
    """I(r; c | y) = sum_y p(y) I(r; c | Y=y)."""
    tot = 0.0
    for v in np.unique(y):
        m = y == v
        if m.sum() > 1:
            tot += m.mean() * mi_mm(r[m], c[m])
    return tot


def route_ids(gates, mask=None, n_clusters=8, seed=0):
    """Discretised route per layer. B-sparse: the selected head set (one of C(H,k)); B-soft: k-means cluster of g.
    gates [n,Lr,H], mask [n,Lr,H] or None -> int [n,Lr]."""
    n, Lr, H = gates.shape
    out = np.zeros((n, Lr), dtype=np.int64)
    if mask is not None:
        out = (mask.astype(np.int64) * (2 ** np.arange(H))).sum(-1)
        return out
    from sklearn.cluster import KMeans
    for l in range(Lr):
        out[:, l] = KMeans(n_clusters=min(n_clusters, n), n_init=5, random_state=seed).fit_predict(gates[:, l])
    return out


def perm_ctx_by_ped(ctx, y, groups, rng):
    """Permute context among pedestrians within each label stratum (whole per-pedestrian c sequences are moved, tiled or
    truncated to the receiving pedestrian's window count), so within-pedestrian structure survives."""
    new = ctx.copy()
    idx_of = {}
    for i, g in enumerate(groups):
        idx_of.setdefault(g, []).append(i)
    for label in np.unique(y):
        peds = [g for g, ix in idx_of.items() if y[ix[0]] == label]
        donors = rng.permutation(len(peds))
        for p, d in zip(peds, donors):
            src = ctx[idx_of[peds[d]]]
            tgt = idx_of[p]
            new[tgt] = np.resize(src, (len(tgt), ctx.shape[1])) if len(src) != len(tgt) else src
    return new


def label_context_mi(route, y, ctx, ctx_names, groups, n_perm=1000, seed=0):
    """Per layer and context variable: I(r;y), I(r;c|y), permutation-null mean / 95th pct / p-value (H3)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    Lr = route.shape[1]
    perms = [perm_ctx_by_ped(ctx, y, groups, rng) for _ in range(n_perm)]
    rows = []
    for l in range(Lr):
        r = route[:, l]
        i_y = mi_mm(r, y)
        for k, name in enumerate(ctx_names):
            if len(np.unique(ctx[:, k])) < 2:
                continue
            obs = cond_mi_mm(r, ctx[:, k], y)
            null = np.array([cond_mi_mm(r, pc[:, k], y) for pc in perms])
            rows.append({'layer': l, 'context': name, 'I(r;y)': i_y, 'I(r;c|y)': obs, 'null_mean': null.mean(),
                         'null_p95': np.percentile(null, 95), 'p_value': (1 + (null >= obs).sum()) / (1 + n_perm),
                         'above_null': bool(obs > np.percentile(null, 95))})
    return pd.DataFrame(rows)


def gate_regression(gates, ctx, y, ctx_names):
    """Per layer and head: partial R^2 of context (all variables, one-hot) in  g ~ c + y.  -> [Lr,H]"""
    n, Lr, H = gates.shape
    cols = []
    for k in range(ctx.shape[1]):
        lv = np.unique(ctx[:, k])
        cols += [(ctx[:, k] == v).astype(float) for v in lv[1:]]
    Xc = np.stack(cols, 1) if cols else np.zeros((n, 0))
    Xy = np.stack([np.ones(n), y.astype(float)], 1)
    Xf = np.concatenate([Xy, Xc], 1)

    def r2(X, t):
        beta = np.linalg.lstsq(X, t, rcond=None)[0]
        v = t.var()
        return 0.0 if v < 1e-12 else 1 - (t - X @ beta).var() / v
    out = np.zeros((Lr, H))
    for l in range(Lr):
        for h in range(H):
            t = gates[:, l, h].astype(float)
            ry = r2(Xy, t)
            out[l, h] = (r2(Xf, t) - ry) / max(1 - ry, 1e-8)
    return out


# ----------------------------------------------------------------------------- model-based helpers
def _gather(ds, idxs):
    items = [ds[int(i)] for i in idxs]
    return torch.stack([b['x'] for b in items]), torch.stack([b['box'] for b in items])   # uint8 [n,T,3,H,W], [n,T,4]


@torch.no_grad()
def _logits(model, XB, device, bs=32, amp=True):
    model.eval()
    X, Bx = XB
    out = []
    for s in range(0, len(X), bs):
        x = prep(X[s:s + bs].to(device))
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            out.append(model(x, box=Bx[s:s + bs].to(device))[0].float().cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- 8.4 head importance (mean ablation)
@torch.no_grad()
def head_importance(model, ds, idxs, device, layers=None, bs=32, amp=True):
    """I[i,l,h] = |u(x_i) - u(x_i; O^{l,h} <- dataset-mean output of head)|.  Returns I [n,L,H], base logits [n]."""
    XB = _gather(ds, idxs)
    X = XB[0]
    L = len(model.blocks)
    layers = list(range(L)) if layers is None else list(layers)
    for l in layers:
        a = model.blocks[l].attn
        a.record = {'sum': torch.zeros(a.h, 197, a.dk, device=device), 'n': 0}
    base = _logits(model, XB, device, bs, amp)
    means = {}
    for l in layers:
        a = model.blocks[l].attn
        means[l] = a.record['sum'] / a.record['n']
        a.record = None
    imp = np.zeros((len(X), L, model.H), dtype=np.float32)
    for l in layers:
        for h in range(model.H):
            model.blocks[l].attn.ablate = (h, means[l][h])
            imp[:, l, h] = np.abs(base - _logits(model, XB, device, bs, amp))
        model.blocks[l].attn.ablate = None
    return imp, base


# ----------------------------------------------------------------------------- 8.5 gradient share
def gradient_norms(model, ds, idxs, device, layers):
    """||grad_{theta^{l,h}} loss_i|| for theta^{l,h} = (Q,K,V rows of head h, W_O columns of head h). -> [n,len(layers),H]"""
    model.eval()
    H, dk = model.H, model.blocks[0].attn.dk
    params = [(model.blocks[l].attn.qkv.weight, model.blocks[l].attn.qkv.bias, model.blocks[l].attn.proj.weight)
              for l in layers]
    flat = [p for t in params for p in t]
    out = np.zeros((len(idxs), len(layers), H), dtype=np.float32)
    for n, i in enumerate(idxs):
        b = ds[int(i)]
        x = prep(b['x'][None].to(device))
        loss = F.binary_cross_entropy_with_logits(model(x, box=b['box'][None].to(device))[0], b['y'].to(device)[None])
        grads = torch.autograd.grad(loss, flat, allow_unused=True)
        for li in range(len(layers)):
            gq, gb, go = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads[3 * li:3 * li + 3], flat[3 * li:3 * li + 3])]
            d = H * dk
            for h in range(H):
                rows = [c * d + h * dk + j for c in range(3) for j in range(dk)]
                sq = gq[rows].pow(2).sum() + gb[rows].pow(2).sum() + go[:, h * dk:(h + 1) * dk].pow(2).sum()
                out[n, li, h] = float(sq.sqrt())
    return out


def gradient_share_entropy(norms, ctx_col, normalise_by_freq=True):
    """q^l_h(c) = sum_{i:c_i=c} ||grad|| / sum_i ||grad||;  returns per-head normalised entropy H(q)/log(#bins) [Lr,H].
    With normalise_by_freq the per-bin MEAN norm is used, so that unbalanced context bins do not look 'specialised'."""
    bins = np.unique(ctx_col)
    q = np.stack([(norms[ctx_col == b].mean(0) if normalise_by_freq else norms[ctx_col == b].sum(0)) for b in bins])
    q = q / (q.sum(0, keepdims=True) + 1e-12)
    return entropy(q, axis=0) / np.log(max(len(bins), 2))


# ----------------------------------------------------------------------------- 8.3 causal route tests
@torch.no_grad()
def _routed_logits(model, tok0_unique, inv, gates_pairs, T, amp, device, box):
    """tok0_unique [U*T,N,d]; inv [P] row of the unique sample for each pair; gates_pairs [P,Lr,H] -> logits, xL."""
    U = tok0_unique.shape[0] // T
    tok = tok0_unique.view(U, T, *tok0_unique.shape[1:])[torch.as_tensor(inv, device=device)].flatten(0, 1)
    g = [gates_pairs[:, r].float() for r in range(gates_pairs.shape[1])]
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.type == 'cuda'):
        logit, info = model.routed_forward(tok, len(inv), T, gates=g, box=box)
    return logit.float().cpu().numpy(), info['xL'].cpu().numpy()


@torch.no_grad()
def swap_experiment(model, ds, own_gates, y, ctx, ctx_names, device, n_pairs=40, n_bins=5, n_ctrl=8, n_cand=40000,
                    bs=32, amp=True, seed=0, tol=0.10):
    """Swap effect  Delta_ij = |u(x_i; g_j) - u(x_i; g_i)|  with pairs matched on route distance ||g_i-g_j||_1, plus the
    magnitude-matched random-gate control (random gate = a real gate with heads permuted per layer, best of n_ctrl candidates
    whose residual-stream change ||X^L(g')-X^L(g_i)|| is closest to the real swap's; matched if within +-tol).
    Returns a DataFrame with one row per pair."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    model.eval()
    n, Lr, H = own_gates.shape
    G = own_gates.reshape(n, -1)
    pairs = {}   # (i,j) -> set of (context, group)
    for k, name in enumerate(ctx_names):
        c = ctx[:, k]
        if len(np.unique(c)) < 2:
            continue
        i = rng.integers(0, n, n_cand)
        j = rng.integers(0, n, n_cand)
        ok = i != j
        i, j = i[ok], j[ok]
        dist = np.abs(G[i] - G[j]).sum(1)
        g1 = (y[i] == y[j]) & (c[i] != c[j])     # same label, different context
        g2 = (c[i] == c[j]) & (y[i] != y[j])     # same context, different label
        if g1.sum() < n_bins or g2.sum() < n_bins:
            continue
        edges = np.quantile(dist[g1 | g2], np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.searchsorted(edges, dist, side='right') - 1, 0, n_bins - 1)
        for bb in range(n_bins):
            m1, m2 = np.where(g1 & (b == bb))[0], np.where(g2 & (b == bb))[0]
            m = min(len(m1), len(m2), n_pairs // n_bins)
            for grp, mm in ((1, m1), (2, m2)):
                for q in rng.choice(mm, m, replace=False) if m > 0 else []:
                    pairs.setdefault((int(i[q]), int(j[q])), set()).add((name, grp))
    plist = sorted(pairs)
    if not plist:
        return pd.DataFrame()
    ids = sorted({i for i, _ in plist})
    X, BX = _gather(ds, ids)
    pos = {i: r for r, i in enumerate(ids)}
    g_t = torch.from_numpy(own_gates).float().to(device)
    res = {k: [] for k in ['u_own', 'u_swap', 'xL_own', 'xL_swap', 'u_ctrl', 'mag_ctrl']}
    T = X.shape[1]
    for s in range(0, len(plist), bs):
        chunk = plist[s:s + bs]
        u_ids = sorted({i for i, _ in chunk})
        tok0 = model.embed_to_L0(prep(X[[pos[i] for i in u_ids]].to(device)))
        inv = [u_ids.index(i) for i, _ in chunk]
        box = BX[[pos[i] for i, _ in chunk]].to(device)   # the pair's own box track (route is swapped, box is not)
        gi = g_t[[i for i, _ in chunk]]
        gj = g_t[[j for _, j in chunk]]
        u_own, x_own = _routed_logits(model, tok0, inv, gi, T, amp, device, box)
        u_sw, x_sw = _routed_logits(model, tok0, inv, gj, T, amp, device, box)
        mag = np.linalg.norm(x_sw - x_own, axis=1)
        best_u, best_mag = np.zeros(len(chunk)), np.full(len(chunk), np.nan)
        best_gap = np.full(len(chunk), np.inf)
        for _ in range(n_ctrl):
            donors = g_t[torch.from_numpy(rng.integers(0, n, len(chunk))).to(device)]
            perm = torch.rand(len(chunk), Lr, H, device=device).argsort(-1)
            gc = torch.gather(donors, 2, perm)
            u_c, x_c = _routed_logits(model, tok0, inv, gc, T, amp, device, box)
            mc = np.linalg.norm(x_c - x_own, axis=1)
            gap = np.abs(mc - mag) / (mag + 1e-8)
            better = gap < best_gap
            best_gap = np.where(better, gap, best_gap)
            best_u = np.where(better, u_c, best_u)
            best_mag = np.where(better, mc, best_mag)
        res['u_own'].append(u_own)
        res['u_swap'].append(u_sw)
        res['xL_own'].append(mag)
        res['xL_swap'].append(best_gap)
        res['u_ctrl'].append(best_u)
        res['mag_ctrl'].append(best_mag)
    r = {k: np.concatenate(v) for k, v in res.items()}
    delta = np.abs(r['u_swap'] - r['u_own'])
    delta_c = np.abs(r['u_ctrl'] - r['u_own'])
    rows = []
    for q, (i, j) in enumerate(plist):
        for name, grp in pairs[(i, j)]:
            rows.append({'context': name, 'group': 'ctx_differs' if grp == 1 else 'label_differs', 'i': i, 'j': j,
                         'delta': delta[q], 'delta_ctrl': delta_c[q], 'mag': r['xL_own'][q], 'matched': r['xL_swap'][q] <= tol})
    return pd.DataFrame(rows)


def swap_summary(df, n_boot=1000, seed=0):
    """Gamma = E[Delta | y_i=y_j, c_i!=c_j] - E[Delta | c_i=c_j, y_i!=y_j] per context; also the 'kept' version
    (only swaps with Delta > Delta' and a matched control) and the same contrast computed on the controls."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for name, d in df.groupby('context'):
        a, b = d[d.group == 'ctx_differs'], d[d.group == 'label_differs']
        if len(a) < 3 or len(b) < 3:
            continue

        def gam(a, b, col):
            return a[col].mean() - b[col].mean()
        boots = [gam(a.iloc[rng.integers(0, len(a), len(a))], b.iloc[rng.integers(0, len(b), len(b))], 'delta')
                 for _ in range(n_boot)]
        ka, kb = a[(a.delta > a.delta_ctrl) & a.matched], b[(b.delta > b.delta_ctrl) & b.matched]
        rows.append({'context': name, 'n_ctx_differs': len(a), 'n_label_differs': len(b),
                     'Gamma': gam(a, b, 'delta'), 'Gamma_lo': np.percentile(boots, 2.5), 'Gamma_hi': np.percentile(boots, 97.5),
                     'Gamma_ctrl': gam(a, b, 'delta_ctrl'),
                     'frac_swap_gt_ctrl': float(((d.delta > d.delta_ctrl) & d.matched).sum() / max(d.matched.sum(), 1)),
                     'Gamma_kept': gam(ka, kb, 'delta') if len(ka) >= 3 and len(kb) >= 3 else np.nan,
                     'n_kept': len(ka) + len(kb)})
    return pd.DataFrame(rows)


def n_routes(H=6, k=3):
    return comb(H, k)
