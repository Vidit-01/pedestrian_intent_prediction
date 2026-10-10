"""One generic Dataset over the sample pkl from build_jaad_samples.py (+ optional cached Swin features).

Input names a model can request (each becomes a key of the batch dict):
  kin              : kinematic vector per frame, built from --kin_feats (standardised with train stats)
                     box   = [x1/W, y1/H, x2/W, y2/H]
                     cwh   = [cx/W, cy/H, w/W, h/H]
                     vel   = frame-to-frame delta of cwh (0 for the first observed frame; observed frames only)
                     speed = one-hot JAAD vehicle action (stopped, moving_slow, moving_fast, decelerating, accelerating)
  speed            : JAAD vehicle action as a scalar 0..4, [T, 1]  (JAAD has no numeric speed; PCPA benchmark does the same)
  box              : [x1/W, y1/H, x2/W, y2/H], [T, 4]
  <modality>       : Swin tokens [T, 64, 1024] float16  (local_rgb, local_flow, global_seg, global_flow, global_rgb)
  <modality>:pool  : the same, averaged over the 64 tokens -> [T, 1024]
"""
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

VISUAL = ['local_rgb', 'local_flow', 'global_seg', 'global_flow', 'global_rgb']
NUM_VEHICLE_ACTIONS = 5


def load_samples(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def norm_boxes(rec):
    wh = rec['img_wh'].astype(np.float32)
    return rec['bbox'] / np.concatenate([wh, wh], 1)[:, None, :]


def kinematic_features(rec, feats):
    b = norm_boxes(rec)
    cwh = np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                    b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)
    parts = {'box': b, 'cwh': cwh,
             'vel': np.concatenate([np.zeros_like(cwh[:, :1]), np.diff(cwh, axis=1)], 1),
             'speed': np.eye(NUM_VEHICLE_ACTIONS, dtype=np.float32)[rec['vehicle_act']]}
    return np.concatenate([parts[f] for f in feats], -1).astype(np.float32)


class FeatureStore:
    """Lazy, mmap-backed access to <feat_root>/<modality>/<vid>.npy written by extract_jaad_visual.py."""

    def __init__(self, root):
        self.root, self.index, self.arrays = root, {}, {}

    def _idx(self, vid):
        if vid not in self.index:
            with open(os.path.join(self.root, 'index', vid + '.json')) as f:
                d = json.load(f)
            self.index[vid] = ({fr: i for i, fr in enumerate(d['global_frames'])},
                               {(p, fr): i for i, (p, fr) in enumerate(d['local_keys'])})
        return self.index[vid]

    def get(self, modality, vid, pid, frames):
        key = (modality, vid)
        if key not in self.arrays:
            self.arrays[key] = np.load(os.path.join(self.root, modality, vid + '.npy'), mmap_mode='r')
        g, l = self._idx(vid)
        rows = [l[(pid, int(f))] for f in frames] if modality.startswith('local') else [g[int(f)] for f in frames]
        return np.asarray(self.arrays[key][rows])


class JAADDataset(Dataset):
    def __init__(self, rec, inputs, kin_feats=('box',), kin_stats=None, feat_root=None):
        self.rec, self.inputs = rec, list(inputs)
        self.labels = rec['label'].astype(np.float32)
        self.kin = None
        if 'kin' in self.inputs:
            self.kin = kinematic_features(rec, kin_feats)
            if kin_stats is not None:
                self.kin = (self.kin - kin_stats[0]) / kin_stats[1]
        self.box = norm_boxes(rec).astype(np.float32)
        self.store = FeatureStore(feat_root) if any(i.split(':')[0] in VISUAL for i in self.inputs) else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        out = {'label': torch.tensor(self.labels[i])}
        for name in self.inputs:
            if name == 'kin':
                out[name] = torch.from_numpy(self.kin[i])
            elif name == 'speed':
                out[name] = torch.from_numpy(self.rec['vehicle_act'][i].astype(np.float32)[:, None])
            elif name == 'box':
                out[name] = torch.from_numpy(self.box[i])
            else:
                mod, _, pool = name.partition(':')
                x = self.store.get(mod, self.rec['vid'][i], self.rec['pid'][i], self.rec['frames'][i])
                out[name] = torch.from_numpy(x.astype(np.float32).mean(1) if pool else x)
        return out


def kin_stats_from(rec, feats):
    """Per-feature mean/std over the TRAIN split only."""
    x = kinematic_features(rec, feats)
    x = x.reshape(-1, x.shape[-1])
    return x.mean(0), x.std(0) + 1e-6


def input_dims(inputs, kin_feats):
    sizes = {'box': 4, 'cwh': 4, 'vel': 4, 'speed': NUM_VEHICLE_ACTIONS}
    dims = {}
    for name in inputs:
        if name == 'kin':
            dims[name] = sum(sizes[f] for f in kin_feats)
        elif name == 'speed':
            dims[name] = 1
        elif name == 'box':
            dims[name] = 4
        else:
            dims[name] = 1024
    return dims
