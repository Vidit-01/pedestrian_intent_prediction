"""Pedestrian Graph+ (Cadena, Qian, Wang, Yang; IEEE T-ITS 2022) - clean-room PyTorch reimplementation.

Written from the authors' public code (github.com/RodrigoGantier/Pedestrian_graph_plus, MIT; read, not executed) because the paper PDF is
paywalled. Layer shapes match the released checkpoints (see `check_against_checkpoint`), ~72 k parameters.

Pipeline:  keypoints [N,C,T,V] -> BatchNorm over (C*V) -> 2 x (adaptive GCN + temporal conv, residual) -> global average pool
           -> sigmoid attention gate (x*att(x)+x) -> dropout -> linear.
Side modules (optional):
  frames : local-context image of the pedestrian (3 or 4 channels incl. a segmentation map), a small 3-layer conv stack; after
           each stage its pooled sigmoid map multiplicatively gates the graph features.
  vel    : ego-vehicle signal [N,2,T] (JAAD: vehicle-action category tiled to 2 channels), a 3-layer Conv1d stack, same gating.
The graph is *learned*: 3 adjacency subsets initialised to the identity (`PA`, L2-normalised per forward), V = 19 joints.
Known quirk kept on purpose for parity with the released code: the stage-1 image gate is computed but not applied
(`x1.mul(...)` result is discarded); `stage1_image_gate=True` applies it.
"""
import math

import numpy as np
import torch
import torch.nn as nn


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    n, k1, k2 = conv.weight.size(0), conv.weight.size(1), conv.weight.size(2)
    nn.init.normal_(conv.weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class UnitTCN(nn.Module):
    def __init__(self, cin, cout, kernel_size=5, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (kernel_size, 1), padding=((kernel_size - 1) // 2, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(cout)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class UnitGCN(nn.Module):
    """Adaptive graph convolution: y = sum_i Conv1x1_i( x A_i ),  A = L2-normalised learnable adjacency (3 subsets)."""

    def __init__(self, cin, cout, A):
        super().__init__()
        self.num_subset = A.shape[0]
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.conv_d = nn.ModuleList([nn.Conv2d(cin, cout, 1) for _ in range(self.num_subset)])
        self.down = nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout)) if cin != cout else nn.Identity()
        self.bn = nn.BatchNorm2d(cout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for c in self.conv_d:
            conv_branch_init(c, self.num_subset)

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.PA / (torch.norm(self.PA, 2, dim=1, keepdim=True) + 1e-4)
        y = 0
        xr = x.reshape(N, C * T, V)
        for i in range(self.num_subset):
            y = y + self.conv_d[i](torch.matmul(xr, A[i]).view(N, C, T, V))
        return torch.relu(self.bn(y) + self.down(x))


class TCNGCNUnit(nn.Module):
    def __init__(self, cin, cout, A, residual=True):
        super().__init__()
        self.gcn1 = UnitGCN(cin, cout, A)
        self.tcn1 = UnitTCN(cout, cout)
        if not residual:
            self.residual = lambda x: 0
        elif cin == cout:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(cin, cout, kernel_size=1)

    def forward(self, x):
        return torch.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


def _cb2d(cin, cout, k):
    return nn.Sequential(nn.Conv2d(cin, cout, k, bias=False), nn.BatchNorm2d(cout), nn.SiLU())


def _cb1d(cin, cout, k):
    return nn.Sequential(nn.Conv1d(cin, cout, k, bias=False), nn.BatchNorm1d(cout), nn.SiLU())


class PedGraphPlus(nn.Module):
    def __init__(self, frames=True, vel=True, seg=False, h3d=False, nodes=19, n_clss=1, stage1_image_gate=False):
        super().__init__()
        self.frames, self.vel, self.stage1_image_gate = frames, vel, stage1_image_gate
        self.ch = 4 if h3d else 3            # (x, y, [depth,] confidence)
        c1, c2 = 32, 64
        i_ch = 4 if seg else 3
        self.data_bn = nn.BatchNorm1d(self.ch * nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(nodes)] * 3, 0)
        if frames:
            self.conv0, self.conv1, self.conv2 = _cb2d(i_ch, c1, 3), _cb2d(c1, c1, 3), _cb2d(c1, c2, 2)
        if vel:
            self.v0, self.v1, self.v2 = _cb1d(2, c1, 3), _cb1d(c1, c1, 3), _cb1d(c1, c2, 2)
        self.l1 = TCNGCNUnit(self.ch, c1, A, residual=False)
        self.l2 = TCNGCNUnit(c1, c2, A)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(nn.SiLU(), nn.Linear(c2, c2, bias=False), nn.BatchNorm1d(c2), nn.Sigmoid())
        self.linear = nn.Linear(c2, n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / n_clss))
        self.pool_sigm_2d = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Sigmoid())
        self.pool_sigm_1d = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Sigmoid())

    def forward(self, kp, frame=None, vel=None):
        N, C, T, V = kp.shape
        kp = self.data_bn(kp.permute(0, 1, 3, 2).reshape(N, C * V, T))
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        f1 = self.conv0(frame) if self.frames else None
        v1 = self.v0(vel) if self.vel else None
        x = self.l1(kp)
        if self.frames:
            f1 = self.conv1(f1)
            if self.stage1_image_gate:
                x = x * self.pool_sigm_2d(f1)
        if self.vel:
            v1 = self.v1(v1)
            x = x * self.pool_sigm_1d(v1).unsqueeze(-1)
        x = self.l2(x)
        if self.frames:
            f1 = self.conv2(f1)
            x = x * self.pool_sigm_2d(f1)
        if self.vel:
            v1 = self.v2(v1)
            x = x * self.pool_sigm_1d(v1).unsqueeze(-1)
        x = self.gap(x).flatten(1)
        x = self.att(x) * x + x
        return self.linear(self.drop(x))


def check_against_checkpoint(path, **kw):
    """Compare parameter names/shapes with a released checkpoint (loaded with weights_only=True)."""
    sd = torch.load(path, map_location='cpu', weights_only=True)
    mine = PedGraphPlus(**kw).state_dict()
    ok = set(sd) == set(mine) and all(sd[k].shape == mine[k].shape for k in sd if k in mine)
    return ok, sorted(set(sd) ^ set(mine)), sum(v.numel() for v in sd.values()), sum(v.numel() for v in mine.values())
