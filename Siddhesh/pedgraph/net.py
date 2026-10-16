"""Configurable Pedestrian-Graph-style network used by the optimisation pipeline (round 2).

`arch='released'` wraps the clean-room reimplementation of the authors' network (model.PedGraphPlus, 70 k parameters) so the
original recipe can be cross-validated with exactly the same protocol as the new architecture.

`arch='net'` (PedGraphNet) keeps the ideas of Pedestrian Graph+ (adaptive-adjacency graph convolution + temporal convolution on the
skeleton, multiplicative gating by side branches) but exposes what the search needs:
  * widths / depth / temporal kernel / dropout of the graph stack,
  * adjacency initialisation: identity (as released) or [identity, skeleton, 2-hop skeleton] (a spatial prior, still learnable),
  * a side branch on per-frame vectors (box trajectory, ego-vehicle action) and an image branch on the local-context crop,
  * fusion by gating (as released), by concatenating the pooled branch features into the classifier, or both.
Everything is fully convolutional over time, so shortened sequences (time-crop augmentation) need no special casing.
"""
import numpy as np
import torch
import torch.nn as nn

from .fastdata import SKELETON
from .model import PedGraphPlus, UnitGCN, UnitTCN, bn_init, conv_init


def skeleton_adjacency(nodes=19, init='identity'):
    eye = np.eye(nodes)
    if init == 'identity':
        return np.stack([eye] * 3, 0)
    A = np.zeros((nodes, nodes))
    for a, b in SKELETON:
        A[a, b] = A[b, a] = 1
    A1 = A / np.maximum(A.sum(0, keepdims=True), 1)           # column-normalised neighbours
    A2 = (A @ A > 0).astype(float) - A - eye
    A2 = np.clip(A2, 0, 1)
    A2 = A2 / np.maximum(A2.sum(0, keepdims=True), 1)         # 2-hop neighbours
    return np.stack([eye, A1, A2], 0)


class Unit(nn.Module):
    """GCN -> TCN with an optional residual connection (kernel size configurable, unlike model.TCNGCNUnit)."""

    def __init__(self, cin, cout, A, kernel, residual):
        super().__init__()
        self.gcn, self.tcn = UnitGCN(cin, cout, A), UnitTCN(cout, cout, kernel_size=kernel)
        if not residual:
            self.res = None
        elif cin == cout:
            self.res = nn.Identity()
        else:
            self.res = UnitTCN(cin, cout, kernel_size=1)

    def forward(self, x):
        y = self.tcn(self.gcn(x))
        return torch.relu(y + (self.res(x) if self.res is not None else 0))


def _cb1d(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm1d(cout), nn.SiLU())


def _cb2d(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(), nn.MaxPool2d(2))


class PedGraphNet(nn.Module):
    def __init__(self, c_in, side_c=0, use_image=False, nodes=19, widths=(32, 64), kernel=5, dropout=0.25,
                 adj_init='identity', fusion='gate', head_hidden=0):
        super().__init__()
        assert fusion in ('gate', 'concat', 'both')
        self.widths, self.side_c, self.use_image, self.fusion = list(widths), side_c, use_image, fusion
        A = skeleton_adjacency(nodes, adj_init)
        self.data_bn = nn.BatchNorm1d(c_in * nodes)
        bn_init(self.data_bn, 1)
        self.units = nn.ModuleList()
        cin = c_in
        for i, w in enumerate(widths):
            self.units.append(Unit(cin, w, A, kernel, residual=i > 0))
            cin = w
        self.side = nn.ModuleList()
        if side_c:
            self.side_bn = nn.BatchNorm1d(side_c)
            cs = side_c
            for w in widths:
                self.side.append(_cb1d(cs, w))
                cs = w
        self.imgs = nn.ModuleList()
        if use_image:
            ci = 3
            for w in widths:
                self.imgs.append(_cb2d(ci, w))
                ci = w
        self.gap = nn.AdaptiveAvgPool2d(1)
        w = widths[-1]
        self.att = nn.Sequential(nn.SiLU(), nn.Linear(w, w, bias=False), nn.BatchNorm1d(w), nn.Sigmoid())
        n_cat = w * (1 + (side_c > 0 and fusion != 'gate') + (use_image and fusion != 'gate'))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_cat, 1) if not head_hidden else nn.Sequential(
            nn.Linear(n_cat, head_hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))

    def forward(self, kp, side=None, img=None):
        N, C, T, V = kp.shape
        x = self.data_bn(kp.permute(0, 1, 3, 2).reshape(N, C * V, T)).view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        s = self.side_bn(side) if self.side_c else None
        im = img
        gate = self.fusion in ('gate', 'both')
        for i, unit in enumerate(self.units):
            x = unit(x)
            if self.side_c:
                s = self.side[i](s)
                if gate:
                    x = x * torch.sigmoid(s.mean(-1))[:, :, None, None]
            if self.use_image:
                im = self.imgs[i](im)
                if gate:
                    x = x * torch.sigmoid(im.mean((2, 3)))[:, :, None, None]
        f = self.gap(x).flatten(1)
        f = self.att(f) * f + f
        if self.fusion != 'gate':
            if self.side_c:
                f = torch.cat([f, s.mean(-1)], 1)
            if self.use_image:
                f = torch.cat([f, im.mean((2, 3))], 1)
        return self.head(self.drop(f))


class Released(nn.Module):
    """model.PedGraphPlus behind the (kp, side, img) interface; side must be the 2-channel tiled ego category."""

    def __init__(self, use_image, use_side):
        super().__init__()
        self.m = PedGraphPlus(frames=use_image, vel=use_side)

    def forward(self, kp, side=None, img=None):
        return self.m(kp, img, side)


def build_model(cfg, c_in, side_c):
    if cfg['arch'] == 'gtranspdm':
        from .gtranspdm import GTransPDM
        return GTransPDM()
    if cfg['arch'] == 'released':
        return Released(cfg['use_image'], side_c > 0)
    return PedGraphNet(c_in, side_c, cfg['use_image'], widths=cfg['widths'], kernel=cfg['kernel'], dropout=cfg['dropout'],
                       adj_init=cfg['adj_init'], fusion=cfg['fusion'], head_hidden=cfg['head_hidden'])
