"""Baseline B: ACIT (Li & Mueller, arXiv:2511.20020), reimplemented in PyTorch from the paper (no official code).

Inputs per sample (T = 16):
  local_rgb, local_flow, global_seg, global_flow : frozen Swin-V2-B final-stage maps [T, 64, 1024]
  speed [T, 1], box [T, 4]
Pipeline:  AVMI(L_RGB, L_OF) -> F_L,  AVMI(G_S, G_OF) -> F_G,  AMMI(S, B) -> F_M   (each [B, T, 256])
           MMFF (inter-modal attention per time step) -> concat [B, T, 768]
           TFA: CLS + positional encoding + 2-layer Transformer encoder (6 heads) -> CLS -> MLP -> logit

Stated in the paper: Swin-V2-B, 256x256 inputs, 8x8x1024 maps, 1x1 conv to 256, positional encodings on the
64 tokens, single-head-formula SA / guided attention (Eq. 1-2) with Norm + residual and learnable alpha, GAP;
AMMI embeddings to 256 + Norm, two MHA units (Eq. 3-4) with Drop&Add, sum, Norm, FFN, Drop&Add;
MMFF Eq. 5-6; TFA 2 layers / 6 heads with post-norm; dropout 0.3; L2 1e-3 on the MLP; BCE; Adam 2e-5;
model size 5.15 M parameters.
NOT stated (our choices, all arguments): AMMI heads (8), FFN widths (256), TFA head size (32, Keras-style key_dim),
alpha init (1.0), learnable positional encodings, MLP hidden (64). These defaults give 5.20 M trainable parameters
(paper: 5.15 M).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathAttention(nn.Module):
    """AVMI for one visual pair: primary (RGB / semantic) self-attention + flow-guided attention (Fig. 2)."""

    def __init__(self, in_ch=1024, d=256, n_tokens=64, alpha_init=1.0):
        super().__init__()
        self.conv_p = nn.Linear(in_ch, d)  # 1x1 conv on a flattened map == per-token linear layer
        self.conv_a = nn.Linear(in_ch, d)
        self.pos_p = nn.Parameter(torch.randn(1, n_tokens, d) * 0.02)
        self.pos_a = nn.Parameter(torch.randn(1, n_tokens, d) * 0.02)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)        # RGB-SA, Eq. 1
        self.q_g, self.k_g, self.v_g = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)  # OF-GA, Eq. 2
        self.norm_sa, self.norm_ga = nn.LayerNorm(d), nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, primary, aux):  # [B, T, 64, C] each
        B, T = primary.shape[:2]
        p = self.conv_p(primary.flatten(0, 1).float()) + self.pos_p
        a = self.conv_a(aux.flatten(0, 1).float()) + self.pos_a
        sa = F.scaled_dot_product_attention(self.q(p), self.k(p), self.v(p))
        ga = F.scaled_dot_product_attention(self.q_g(p), self.k_g(a), self.v_g(a))
        x = p + self.norm_sa(sa) + self.alpha * self.norm_ga(ga)
        return x.mean(1).view(B, T, -1)  # GAP over the 8x8 tokens


class MotionInteraction(nn.Module):
    """AMMI: bidirectional cross-attention between ego speed and pedestrian box over the time steps."""

    def __init__(self, d=256, heads=8, ffn=512, dropout=0.3):
        super().__init__()
        self.emb_s, self.emb_b = nn.Linear(1, d), nn.Linear(4, d)
        self.norm_s, self.norm_b = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha_s = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)  # Q from S, K/V from B
        self.mha_b = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)  # Q from B, K/V from S
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Linear(ffn, d))

    def forward(self, s, b):  # [B, T, 1], [B, T, 4]
        xs, xb = self.norm_s(self.emb_s(s)), self.norm_b(self.emb_b(b))
        ys = xs + self.drop(self.mha_s(xs, xb, xb, need_weights=False)[0])
        yb = xb + self.drop(self.mha_b(xb, xs, xs, need_weights=False)[0])
        h = self.norm(ys + yb)
        return h + self.drop(self.ffn(h))


class ModalityFusion(nn.Module):
    """MMFF: at each time step, modality i attends over modalities j (Eq. 5-6); outputs concatenated."""

    def __init__(self, d=256, n=3):
        super().__init__()
        self.q = nn.ModuleList(nn.Linear(d, d) for _ in range(n))
        self.k = nn.ModuleList(nn.Linear(d, d) for _ in range(n))
        self.v = nn.ModuleList(nn.Linear(d, d) for _ in range(n))
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, feats):  # list of n tensors [B, T, d]
        Q = torch.stack([f(x) for f, x in zip(self.q, feats)], 2)  # [B, T, n, d]
        K = torch.stack([f(x) for f, x in zip(self.k, feats)], 2)
        V = torch.stack([f(x) for f, x in zip(self.v, feats)], 2)
        attn = torch.softmax(Q @ K.transpose(-1, -2) * self.scale, -1)  # alpha_ij, [B, T, n, n]
        return (attn @ V).flatten(2)  # [B, T, n*d]


class TemporalLayer(nn.Module):
    """Post-norm Transformer encoder layer (MHSA -> Add&Norm -> FFN -> Add&Norm), Eq. 7-8."""

    def __init__(self, d=768, heads=6, head_dim=64, ffn=256, dropout=0.3):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.qkv = nn.Linear(d, 3 * heads * head_dim)
        self.out = nn.Linear(heads * head_dim, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Linear(ffn, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, -1)
        x = self.norm1(x + self.drop(self.out(a)))
        return self.norm2(x + self.drop(self.ffn(x)))


class ACIT(nn.Module):
    inputs = ['local_rgb', 'local_flow', 'global_seg', 'global_flow', 'speed', 'box']

    def __init__(self, T=16, in_ch=1024, d=256, ammi_heads=8, ammi_ffn=256, tfa_layers=2, tfa_heads=6,
                 tfa_head_dim=32, tfa_ffn=256, mlp_hidden=64, dropout=0.3, l2=1e-3, alpha_init=1.0):
        super().__init__()
        self.l2 = l2
        self.avmi_local = DualPathAttention(in_ch, d, alpha_init=alpha_init)
        self.avmi_global = DualPathAttention(in_ch, d, alpha_init=alpha_init)
        self.ammi = MotionInteraction(d, ammi_heads, ammi_ffn, dropout)
        self.mmff = ModalityFusion(d, 3)
        D = 3 * d
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.randn(1, T + 1, D) * 0.02)
        self.tfa = nn.ModuleList(TemporalLayer(D, tfa_heads, tfa_head_dim, tfa_ffn, dropout) for _ in range(tfa_layers))
        self.mlp = nn.Sequential(nn.Linear(D, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))

    def forward(self, batch):
        f_l = self.avmi_local(batch['local_rgb'], batch['local_flow'])
        f_g = self.avmi_global(batch['global_seg'], batch['global_flow'])
        f_m = self.ammi(batch['speed'].float(), batch['box'].float())
        x = self.mmff([f_l, f_g, f_m])
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        for layer in self.tfa:
            x = layer(x)
        return self.mlp(x[:, 0]).squeeze(-1)

    def reg_loss(self):
        """Keras-style L2 (l2 * sum(w^2)) on the MLP weight matrices, as in the paper."""
        return self.l2 * sum((m.weight ** 2).sum() for m in self.mlp if isinstance(m, nn.Linear))
