"""PedViViT-style pedestrian-intent model (VIT-Ped, arXiv 2601.01989) with optional head routing. Spec: idea.md sections 2-4, 6.

PedViViT = ViViT factorised encoder (per-frame spatial ViT -> temporal transformer over per-frame CLS tokens) on the
pedestrian local-context crop, + a small non-visual transformer on delta-encoded bounding boxes (their 'Ours 6': 2 layers,
4 heads), fused by concatenation + FFN into one sigmoid unit. Unspecified in the paper and chosen here: ViT-S/16 spatial encoder
(ImageNet-pretrained), 2-layer temporal transformer, box embedding width 64, FFN width 128. Ego speed is NOT an input
(it is one of the analysed contexts). The routing of idea.md acts only inside the spatial ViT.

variants:  dense  - every head contributes equally
           soft   - B-soft:   g = H * softmax(s / tau)
           sparse - B-sparse: top-k heads, g = H * softmax over the selected logits (noise only picks the set)
           film   - unconstrained FiLM on the concatenated head space (gamma = 1 + W z, beta = W' z), control for Prop. 2
Training-stability additions (round 2): stochastic depth in the spatial ViT, optional freezing of the first blocks, optional
box-only control (no images at all). None of them changes the function at initialisation (Prop. 1 still holds in eval mode).
Routed layers are blocks L0..L-1 (0-based), i.e. 'blocks 7-12' for L0 = 6. One route per clip, from
z = LN(frame-averaged CLS token after block L0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """MHA written as a sum of per-head terms: Concat(g_h O_h) W_O = sum_h g_h O_h W_O^(h) (+ bias)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.record = None   # {'sum': [h,N,dk], 'n': int} -> running mean of per-head outputs (analysis only)
        self.ablate = None   # (head, mean[h,N,dk]) -> mean-ablation of one head (analysis only)

    def forward(self, x, gate=None, film=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)  # [B, h, N, dk]
        if self.record is not None:
            self.record['sum'] += o.float().sum(0)
            self.record['n'] += B
        if self.ablate is not None:
            hh, mean = self.ablate
            o = o.clone()
            o[:, hh] = mean.to(o.dtype)
        if gate is not None:
            o = o * gate.to(o.dtype)[:, :, None, None]
        o = o.transpose(1, 2).reshape(B, N, D)
        if film is not None:
            o = o * film[0][:, None, :].to(o.dtype) + film[1][:, None, :].to(o.dtype)
        return self.proj(o)


class Mlp(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    """Stochastic depth: drops the whole residual branch per sample (training only)."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(d, eps=1e-6), Attention(d, heads)
        self.norm2, self.mlp = nn.LayerNorm(d, eps=1e-6), Mlp(d, d * mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, gate=None, film=None):
        x = x + self.drop_path(self.attn(self.norm1(x), gate, film))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, d, patch=16):
        super().__init__()
        self.proj = nn.Conv2d(3, d, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Router(nn.Module):
    """s = W z + b (zero-init => uniform gates, Prop. 1). B-sparse also has a noise net (training only)."""

    def __init__(self, d, heads, k, sparse):
        super().__init__()
        self.k = k
        self.w = nn.Linear(d, heads)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)
        self.noise = nn.Linear(d, heads) if sparse else None
        if self.noise is not None:
            nn.init.zeros_(self.noise.weight)
            nn.init.zeros_(self.noise.bias)

    def gate(self, z, tau, sparse_on):
        """-> g [B,H] (sum = H), pi [B,H] (full softmax), mask [B,H] bool or None, s [B,H] logits."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            s = self.w(z)
            H = s.shape[-1]
            pi = F.softmax(s / tau, -1)
            if self.noise is None or not sparse_on:
                return H * pi, pi, None, s
            sn = s
            if self.training:
                sn = s + torch.randn_like(s) * F.softplus(self.noise(z))
            idx = sn.topk(self.k, -1).indices
            mask = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True)
            g = H * F.softmax((s / tau).masked_fill(~mask, float('-inf')), -1)
            return g, pi, mask, s


class RoutedViT(nn.Module):
    def __init__(self, variant='dense', d=384, depth=12, heads=6, L0=6, k=3, T=8, img=224, patch=16,
                 t_layers=2, t_dropout=0.1, use_box=True, d_nv=64, drop_path=0.0, freeze_blocks=0, box_only=False,
                 box_drop=0.0):
        super().__init__()
        assert variant in ('dense', 'soft', 'sparse', 'film')
        assert 0 <= freeze_blocks <= L0, 'only dense blocks (before the routed ones) can be frozen'
        assert not box_only or (variant == 'dense' and use_box), 'box_only is a dense control that needs the box branch'
        self.variant, self.d, self.H, self.L0, self.k, self.T = variant, d, heads, L0, k, T
        self.freeze_blocks, self.box_only, self.box_drop = freeze_blocks, box_only, box_drop
        n_tok = (img // patch) ** 2 + 1
        # shared parameters first, so a fixed seed gives identical shared weights for every variant
        # (stochastic-depth rate grows linearly with depth; frozen blocks are never dropped)
        dp = [0.0 if i < freeze_blocks else drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.patch_embed = PatchEmbed(d, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tok, d))
        self.blocks = nn.ModuleList([Block(d, heads, drop_path=dp[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.t_cls = nn.Parameter(torch.zeros(1, 1, d))
        self.t_pos = nn.Parameter(torch.zeros(1, T + 1, d))
        nn.init.trunc_normal_(self.t_pos, std=0.02)
        nn.init.trunc_normal_(self.t_cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=t_dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, t_layers, enable_nested_tensor=False)
        self.use_box = use_box
        if use_box:   # non-visual branch: delta-encoded boxes -> 2-layer / 4-head transformer -> mean pool
            self.nv_in = nn.Sequential(nn.Linear(4, d_nv), nn.LayerNorm(d_nv))
            self.nv_pos = nn.Parameter(torch.zeros(1, T, d_nv))
            nn.init.trunc_normal_(self.nv_pos, std=0.02)
            nv_layer = nn.TransformerEncoderLayer(d_nv, 4, dim_feedforward=2 * d_nv, dropout=t_dropout, activation='gelu',
                                                  batch_first=True, norm_first=True)
            self.nv = nn.TransformerEncoder(nv_layer, 2, enable_nested_tensor=False)
        d_f = (0 if box_only else d) + (d_nv if use_box else 0)
        self.fuse = nn.Sequential(nn.LayerNorm(d_f), nn.Linear(d_f, 128), nn.GELU(), nn.Dropout(t_dropout), nn.Linear(128, 1))
        # routing parameters (variant specific), created last
        n_r = depth - L0
        if variant != 'dense':
            self.z_norm = nn.LayerNorm(d)
        if variant in ('soft', 'sparse'):
            self.routers = nn.ModuleList([Router(d, heads, k, variant == 'sparse') for _ in range(n_r)])
        if variant == 'film':
            self.film = nn.ModuleList([nn.Linear(d, 2 * d) for _ in range(n_r)])
            for m in self.film:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        if freeze_blocks:
            for m in (self.patch_embed, self.blocks[:freeze_blocks]):
                for p in m.parameters():
                    p.requires_grad_(False)
            self.cls_token.requires_grad_(False)
            self.pos_embed.requires_grad_(False)

    def load_pretrained(self, name='vit_small_patch16_224.augreg_in21k_ft_in1k'):
        import timm
        sd = timm.create_model(name, pretrained=True).state_dict()
        sd = {k: v for k, v in sd.items() if not k.startswith(('head', 'fc_norm'))}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        bad = [m for m in missing if m.split('.')[0] in ('patch_embed', 'cls_token', 'pos_embed', 'blocks', 'norm')]
        assert not bad and not unexpected, f'pretrained load mismatch: missing={bad} unexpected={unexpected}'

    @property
    def n_routed(self):
        return len(self.blocks) - self.L0

    def embed_to_L0(self, x):
        """x [B,T,3,H,W] -> tokens after the last dense block, [B*T, N, d]."""
        nf = self.freeze_blocks
        with torch.set_grad_enabled(torch.is_grad_enabled() and nf == 0):   # frozen prefix: no activations kept for backward
            tok = self.patch_embed(x.flatten(0, 1))
            tok = torch.cat([self.cls_token.expand(tok.shape[0], -1, -1), tok], 1) + self.pos_embed
            for blk in self.blocks[:nf]:
                tok = blk(tok)
        for blk in self.blocks[nf:self.L0]:
            tok = blk(tok)
        return tok

    def routed_forward(self, tok, B, T, tau=1.0, sparse_on=True, gates=None, box=None):
        """tok: [B*T,N,d] after block L0. gates: optional list of [B,H] overriding the router (swap tests)."""
        zero = tok.new_zeros(()).float()
        info = {'gates': [], 'pi': [], 'mask': [], 'bal': zero, 'zloss': zero}
        z = None
        if self.variant != 'dense':
            with torch.autocast(device_type=tok.device.type, enabled=False):
                z = self.z_norm(tok[:, 0].float().view(B, T, -1).mean(1))
            info['z'] = z
        for li in range(self.L0, len(self.blocks)):
            blk, r = self.blocks[li], li - self.L0
            gate = film = None
            if self.variant == 'film':
                gb = self.film[r](z)
                film = ((1 + gb[:, :self.d]).repeat_interleave(T, 0), gb[:, self.d:].repeat_interleave(T, 0))
            elif self.variant in ('soft', 'sparse'):
                if gates is not None:
                    g = gates[r]
                else:
                    g, pi, mask, s = self.routers[r].gate(z, tau, sparse_on)
                    info['pi'].append(pi)
                    info['mask'].append(mask)
                    info['zloss'] = info['zloss'] + (torch.logsumexp(s, -1) ** 2).mean()
                    if mask is not None:   # B-sparse: H * sum_h f_h P_h
                        info['bal'] = info['bal'] + self.H * ((mask.float().mean(0) / self.k) * pi.mean(0)).sum()
                    else:                  # B-soft: CV^2 of the per-head load
                        load = g.sum(0)
                        info['bal'] = info['bal'] + load.var(unbiased=False) / (load.mean() ** 2 + 1e-8)
                info['gates'].append(g)
                gate = g.repeat_interleave(T, 0)
            tok = blk(tok, gate, film)
        info['xL'] = tok[:, 0].float().view(B, T, -1).mean(1)   # residual-stream CLS at the last block (pre final LN)
        cls = self.norm(tok)[:, 0].view(B, T, -1)
        h = torch.cat([self.t_cls.expand(B, -1, -1).to(cls.dtype), cls], 1) + self.t_pos.to(cls.dtype)
        f = self.temporal(h)[:, 0]
        if self.use_box:
            fb = self.nv(self.nv_in(box.to(f.dtype)) + self.nv_pos.to(f.dtype)).mean(1)
            if self.training and self.box_drop > 0:   # modality dropout: stops the visual branch from being ignored
                fb = fb * (torch.rand(fb.shape[0], 1, device=fb.device) >= self.box_drop).to(fb.dtype)
            f = torch.cat([f, fb], 1)
        return self.fuse(f).squeeze(-1), info

    def box_only_forward(self, box):
        """Control: the non-visual branch alone (no pixels). Tells how much of the signal is bounding-box dynamics."""
        z = box.new_zeros(()).float()
        h = self.nv_in(box)
        f = self.nv(h + self.nv_pos.to(h.dtype)).mean(1)
        return self.fuse(f).squeeze(-1), {'gates': [], 'pi': [], 'mask': [], 'bal': z, 'zloss': z}

    def forward(self, x, box=None, tau=1.0, sparse_on=True, gates=None):
        if self.box_only:
            return self.box_only_forward(box)
        B, T = x.shape[:2]
        return self.routed_forward(self.embed_to_L0(x), B, T, tau, sparse_on, gates, box)


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    rout = sum(p.numel() for n, p in model.named_parameters() if n.startswith(('routers', 'film', 'z_norm')))
    return tot, rout
