"""GTransPDM reimplementation (no public code exists; written from the paper).

    Xie, Lin, Zheng, Gong, Lopez. "GTransPDM: A Graph-embedded Transformer with Positional Decoupling for Pedestrian Crossing
    Intention Prediction", IEEE Signal Processing Letters 2025, arXiv 2409.20223 (v2, May 2025).   [literature map entry 5.3]

Taken from the paper (Sec. III-IV, training setup):
  * input: T = 16 observed frames, TTE 30-60, overlap 0.8 on JAAD_all (default split) = this repo's PCPA samples (jaad_all.pkl);
  * N = 20 keypoints, coordinates relative to the box top-left corner, + confidence;  box centre (x,y) and area A per frame;
    ego motion: S (per-frame; on JAAD the 5 ego-motion states replace speed) and a global acceleration
    Acc = (s_T - s_0) * FPS / (3.6 * T);
  * Positional Decoupling Module: P_t = (dx_t - dx_{t-1}, dy_t - dy_{t-1}, R_t) with R_t = (A_t / A_{t-1} - 1) * 100, plus
    displacement D_t = (x_t - x_0, y_t - y_0) and velocity V_t = (x_t - x_{t-1}, y_t - y_{t-1});  dx, dy are offsets to the
    two reference lines A=(0,H) - B=(W/2, Y_min) - C=(W,H);
  * skeleton encoder: 4 GCN layers, d_hid = 64, A_hat = D^-1/2 (A + I) D^-1/2, learnable edge weights E_l (Hadamard),
    layers 1-3 with a Conv2D residual branch, flatten T x (N d_hid) -> FC -> 64;
  * position / ego encoders: one FC per feature, concatenate, FC -> 64;  fusion: concat (3 x 64) -> FC -> 64;
  * Transformer encoder: 4 layers, 4 heads, hidden 64, dropout 0.1, temporal positional encoding; flatten -> FC -> softmax;
  * training: AdamW, lr 5e-5 constant, batch 64, 32 epochs, weight decay 1e-4, cross-entropy (no class weighting, no augmentation).

NOT specified in the paper, chosen here (each is a deviation or assumption to report):
  * reference point (x_l, y_l): the paper only says the offsets to the lines are used. Here: the point of the polyline (line AB for
    x < W/2, line BC otherwise) at the pedestrian's row for dx, and at the pedestrian's column for dy; Y_min is taken over the
    pooled train+val windows;
  * Conv2D kernel (1x1), edge list of the skeleton (COCO limbs + neck, pelvis, body-centre chain), which joints are averaged for the
    three extra points (neck = shoulders, hip = hips, body centre = neck/hip midpoint), feed-forward width of the Transformer (256),
    sinusoidal positional encoding, ReLU;
  * BatchNorm on the keypoint and on the position / ego inputs (the paper writes BatchNorm(K_i) for keypoints only; raw pixel
    offsets would otherwise enter FC layers unscaled);
  * poses come from ViTPose (the paper uses AlphaPose); softmax over 2 classes is replaced by one logit with BCE (equivalent family);
  * on JAAD the five ego states are fed as the scalar category 0..4 (the paper does not say how).
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .fastdata import SKELETON

N_NODES = 20            # 17 COCO + neck (17) + pelvis/hip (18) + body centre (19)
CENTER = 19
FPS = 30.0


def edges20():
    e = [(a, b) for a, b in SKELETON if (a, b) != (17, 18)] + [(17, CENTER), (CENTER, 18)]
    return e


def normalized_adjacency(n=N_NODES):
    A = np.eye(n)
    for a, b in edges20():
        A[a, b] = A[b, a] = 1
    d = A.sum(1)
    return (A / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]).astype(np.float32)


# --------------------------------------------------------------------------- features
def build_batch_g(D, rows, cfg, train=False, **_):
    """-> pose [B,3,T,20] (x - x1, y - y1, score), side [B,9,T] = [P(3), D(2), V(2), S, Acc], None.  The paper uses no augmentation."""
    rows_t = torch.as_tensor(rows, device=D.device)
    kp, box, wh = D.kp[rows_t], D.box[rows_t], D.wh[rows_t]
    B, T = kp.shape[:2]
    neck, hip = kp[:, :, 17], kp[:, :, 18]
    center = torch.cat([(neck[..., :2] + hip[..., :2]) / 2, torch.minimum(neck[..., 2:], hip[..., 2:])], -1)
    kp = torch.cat([kp, center[:, :, None]], 2)                                        # [B,T,20,3]
    rel = torch.stack([kp[..., 0] - box[..., None, 0], kp[..., 1] - box[..., None, 1], kp[..., 2]], -1)
    pose = rel.permute(0, 3, 1, 2).contiguous()                                        # [B,3,T,20]
    x, y = (box[..., 0] + box[..., 2]) / 2, (box[..., 1] + box[..., 3]) / 2            # [B,T]
    area = ((box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])).clamp(min=1.0)
    W, H = wh[:, 0:1], wh[:, 1:2]
    span = (H - D.ymin).clamp(min=1.0)
    left = x < W / 2
    y_line = torch.where(left, H - span * (x / (W / 2)), H - span * ((W - x) / (W / 2)))
    x_line = torch.where(left, (W / 2) * (H - y) / span, W - (W / 2) * (H - y) / span)
    dx, dy = x - x_line, y - y_line
    zero = torch.zeros(B, 1, device=D.device)
    diff = lambda a: torch.cat([zero, a[:, 1:] - a[:, :-1]], 1)
    R = torch.cat([zero, (area[:, 1:] / area[:, :-1] - 1) * 100], 1)
    P = torch.stack([diff(dx), diff(dy), R], -1)
    Dis = torch.stack([x - x[:, :1], y - y[:, :1]], -1)
    V = torch.stack([diff(x), diff(y)], -1)
    S = D.veh[rows_t].float()                                                          # [B,T]
    acc = ((S[:, -1] - S[:, 0]) * FPS / (3.6 * T))[:, None].expand(-1, T)
    side = torch.cat([P, Dis, V, S[..., None], acc[..., None]], -1).permute(0, 2, 1).contiguous()   # [B,9,T]
    return pose, side, None


# --------------------------------------------------------------------------- model
class GTransPDM(nn.Module):
    def __init__(self, T=16, d=64, gcn_layers=4, t_layers=4, heads=4, dropout=0.1, ff=256, nodes=N_NODES):
        super().__init__()
        self.T, self.nodes, self.d = T, nodes, d
        self.register_buffer('A_hat', torch.from_numpy(normalized_adjacency(nodes)))
        self.bn_kp = nn.BatchNorm1d(3 * nodes)
        self.E = nn.ParameterList([nn.Parameter(torch.ones(nodes, nodes)) for _ in range(gcn_layers)])
        self.W = nn.ModuleList([nn.Linear(3 if i == 0 else d, d) for i in range(gcn_layers)])
        self.Wc = nn.ModuleList([nn.Conv2d(d, d, 1) for _ in range(gcn_layers - 1)])
        self.kp_fc = nn.Linear(nodes * d, d)
        self.bn_side = nn.BatchNorm1d(9)
        self.p_fc, self.d_fc, self.v_fc = nn.Linear(3, d), nn.Linear(2, d), nn.Linear(2, d)
        self.pe_fc = nn.Linear(3 * d, d)
        self.s_fc, self.a_fc = nn.Linear(1, d), nn.Linear(1, d)
        self.ev_fc = nn.Linear(2 * d, d)
        self.fuse = nn.Linear(3 * d, d)
        pe = torch.zeros(1, T, d)
        pos, div = torch.arange(T)[:, None], torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ff, dropout=dropout, activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, t_layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(T * d, 1)

    def forward(self, kp, side=None, img=None):
        B, C, T, N = kp.shape
        x = self.bn_kp(kp.permute(0, 1, 3, 2).reshape(B, C * N, T)).view(B, C, N, T).permute(0, 3, 2, 1)   # [B,T,N,3]
        h = x
        for l in range(len(self.W)):
            agg = torch.einsum('nm,btmc->btnc', self.E[l] * self.A_hat, h)
            z = self.W[l](agg)
            if l > 0:
                z = z + self.Wc[l - 1](h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            h = torch.relu(z)
        x_ke = self.kp_fc(h.reshape(B, T, N * self.d))
        s = self.bn_side(side).permute(0, 2, 1)                                                           # [B,T,9]
        x_pe = self.pe_fc(torch.cat([self.p_fc(s[..., 0:3]), self.d_fc(s[..., 3:5]), self.v_fc(s[..., 5:7])], -1))
        x_ev = self.ev_fc(torch.cat([self.s_fc(s[..., 7:8]), self.a_fc(s[..., 8:9])], -1))
        f = self.fuse(torch.cat([x_pe, x_ev, x_ke], -1)) + self.pe[:, :T]
        return self.head(self.drop(self.encoder(f).reshape(B, -1)))
