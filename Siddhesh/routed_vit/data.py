"""JAAD data for the routed-ViT study: PCPA-protocol samples (+ annotated context), pedestrian-crop cache, Dataset.

Samples come from the repo's build_jaad_samples.build (obs 16 frames, TTE 30-60, stride 3, official video split).
Each clip uses T=8 of the 16 observed frames (odd indices 1,3,...,15: last observed frame included).
Context c (analysis only, never a model input) is read from the JAAD annotations; see CTX_NAMES.
"""
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FRAME_IDX = tuple(range(1, 16, 2))
CTX_NAMES = ['signalized', 'intersection', 'designated', 'group_size', 'num_lanes', 'ego_act', 'traffic_light',
             'ped_crossing']
CTX_LEVELS = {'signalized': ['n/a', 'NS', 'S'], 'intersection': ['no', 'yes'], 'designated': ['ND', 'D'],
              'group_size': ['1', '2', '3+'], 'num_lanes': ['1', '2', '3', '4+'],
              'ego_act': ['stopped', 'slow', 'fast', 'decel', 'accel'], 'traffic_light': ['n/a', 'red', 'green'],
              'ped_crossing': ['no', 'yes']}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# ----------------------------------------------------------------------------- samples + context
def build_samples(repo_root, jaad_root, out_pkl, sample_type='beh'):
    """PCPA samples via the repo's builder, plus per-window context array rec['ctx'] [n, len(CTX_NAMES)] (int)."""
    sys.path.insert(0, repo_root)
    from build_jaad_samples import build
    data = build(jaad_root, sample_type)
    from jaad_data import JAAD   # build() put jaad_root on sys.path
    db = JAAD(data_path=jaad_root).generate_database()
    for split in ['train', 'val', 'test']:
        rec = data[split]
        ctx = np.zeros((len(rec['label']), len(CTX_NAMES)), dtype=np.int64)
        for i in range(len(rec['label'])):
            vid, pid = rec['vid'][i], rec['pid'][i]
            a = db[vid]['ped_annotations'][pid].get('attributes', {})
            t = db[vid]['traffic_annotations'].get(int(rec['frames'][i][-1]), {})
            ctx[i] = [int(a.get('signalized', 0)), int(a.get('intersection', 0)), int(a.get('designated', 0)),
                      min(max(int(a.get('group_size', 1)), 1), 3) - 1, min(max(int(a.get('num_lanes', 2)), 1), 4) - 1,
                      int(rec['vehicle_act'][i][-1]), int(t.get('traffic_light', 0)), int(t.get('ped_crossing', 0))]
        rec['ctx'] = ctx
    data['meta']['ctx_names'] = CTX_NAMES
    data['meta']['frame_idx'] = list(FRAME_IDX)
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)
    return data


def load_samples(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def groups_of(rec):
    return np.array([f'{v}/{p}' for v, p in zip(rec['vid'], rec['pid'])])


# ----------------------------------------------------------------------------- crop cache
def square_crop(img, box, scale, size):
    import cv2
    x1, y1, x2, y2 = [float(b) for b in box]
    s = max(max(x2 - x1, y2 - y1) * scale, 16.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    l, t = int(round(cx - s / 2)), int(round(cy - s / 2))
    side = int(round(s))
    H, W = img.shape[:2]
    patch = np.zeros((side, side, 3), np.uint8)
    xa, ya, xb, yb = max(l, 0), max(t, 0), min(l + side, W), min(t + side, H)
    if xb > xa and yb > ya:
        patch[ya - t:yb - t, xa - l:xb - l] = img[ya:yb, xa:xb]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)


def _crop_worker(job):
    import cv2
    clips_dir, out_dir, vid, items, size, scale = job
    npy, js = os.path.join(out_dir, vid + '.npy'), os.path.join(out_dir, vid + '.json')
    if os.path.exists(npy) and os.path.exists(js):
        return vid, 'skip'
    by_frame = {}
    for pid, fr, box in items:
        by_frame.setdefault(fr, []).append((pid, box))
    last = max(by_frame)
    cap = cv2.VideoCapture(os.path.join(clips_dir, vid + '.mp4'))
    keys, crops = [], []
    for i in range(last + 1):
        if not cap.grab():
            raise RuntimeError(f'{vid}: could not read frame {i} (need up to {last})')
        if i in by_frame:
            ok, bgr = cap.retrieve()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for pid, box in by_frame[i]:
                keys.append([pid, i])
                crops.append(square_crop(rgb, box, scale, size))
    cap.release()
    np.save(npy + '.tmp.npy', np.stack(crops))
    os.replace(npy + '.tmp.npy', npy)
    with open(js, 'w') as f:
        json.dump(keys, f)
    return vid, len(keys)


def build_crop_cache(data, clips_dir, out_dir, size=224, scale=1.5, frame_idx=FRAME_IDX, workers=4):
    """One uint8 [n,size,size,3] npy + key json per video, containing exactly the (pid, frame) crops the clips use."""
    os.makedirs(out_dir, exist_ok=True)
    need = {}
    for split in ['train', 'val', 'test']:
        rec = data[split]
        for i in range(len(rec['label'])):
            d = need.setdefault(rec['vid'][i], {})
            for j in frame_idx:
                d[(rec['pid'][i], int(rec['frames'][i][j]))] = rec['bbox'][i][j]
    jobs = [(clips_dir, out_dir, v, [(p, f, b) for (p, f), b in d.items()], size, scale) for v, d in need.items()]
    done = 0
    with Pool(workers) as pool:
        for vid, n in pool.imap_unordered(_crop_worker, jobs):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f'  crops {done}/{len(jobs)} videos ({vid}: {n})', flush=True)
    return out_dir


# ----------------------------------------------------------------------------- dataset
class CropDataset(Dataset):
    """Clips of a record `rec`. `indices` restricts it to some rows (cross-validation folds of the pooled train+val record);
    the returned 'idx' is always the row in `rec`, so predictions can be mapped back. images=False skips the pixels (box-only control)."""

    def __init__(self, rec, root, frame_idx=FRAME_IDX, indices=None, images=True):
        self.rec, self.root, self.fi, self.images = rec, root, list(frame_idx), images
        self.rows = np.arange(len(rec['label'])) if indices is None else np.asarray(indices)
        self.arrays, self.index = {}, {}

    def __len__(self):
        return len(self.rows)

    def _load(self, vid):
        if vid not in self.arrays:
            self.arrays[vid] = np.load(os.path.join(self.root, vid + '.npy'), mmap_mode='r')
            with open(os.path.join(self.root, vid + '.json')) as f:
                self.index[vid] = {(p, int(fr)): r for r, (p, fr) in enumerate(json.load(f))}
        return self.arrays[vid], self.index[vid]

    def __getitem__(self, k):
        i = int(self.rows[k])
        vid, pid = self.rec['vid'][i], self.rec['pid'][i]
        if self.images:
            arr, idx = self._load(vid)
            rows = [idx[(pid, int(self.rec['frames'][i][j]))] for j in self.fi]
            x = torch.from_numpy(np.ascontiguousarray(arr[rows])).permute(0, 3, 1, 2)   # [T,3,H,W] uint8
        else:
            x = torch.zeros(len(self.fi), 3, 1, 1, dtype=torch.uint8)
        wh = np.asarray(self.rec['img_wh'][i], dtype=np.float32)
        bb = self.rec['bbox'][i][self.fi] / np.concatenate([wh, wh])
        box = torch.from_numpy(((bb - bb[0]) * 5.0).astype(np.float32))   # delta-encoded vs the first frame (PedViViT)
        return {'x': x, 'box': box, 'y': torch.tensor(float(self.rec['label'][i])), 'idx': i}


def _geo_aug(x, strength, gen=None):
    """Same random zoom (0.9-1.1) / shift (+-6%) / rotation (+-5 deg) for all frames of a clip (x: [B,T,3,H,W] float)."""
    B, T, C, H, W = x.shape
    dev = x.device
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, device=dev, generator=gen)
    sc = 1.0 + strength * u(-0.1, 0.1)
    ang = strength * u(-5.0, 5.0) * 3.14159265 / 180
    tx, ty = strength * u(-0.12, 0.12), strength * u(-0.12, 0.12)    # normalised coords: 0.12 = 6% of the width
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)   # [B,2,3]
    grid = F.affine_grid(theta, (B, T * C, H, W), align_corners=False)
    out = F.grid_sample(x.reshape(B, T * C, H, W), grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return out.reshape(B, T, C, H, W)


def prep(x, train=False, gen=None, geo=1.0):
    """uint8 [B,T,3,H,W] (on the model device) -> normalised float.
    Train aug: hflip + brightness/contrast per clip, plus a clip-consistent small affine jitter (strength `geo`, 0 disables)."""
    x = x.float() / 255.0
    if train:
        B = x.shape[0]
        flip = torch.rand(B, device=x.device, generator=gen) < 0.5
        x = torch.where(flip[:, None, None, None, None], x.flip(-1), x)
        b = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device=x.device, generator=gen)
        c = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device=x.device, generator=gen)
        m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        x = ((x - m) * c + m) * b
        if geo > 0:
            x = _geo_aug(x, geo, gen)
        x = x.clamp(0, 1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def context_table(data):
    """Tidy table: one row per context variable, one column per split with 'level:count' strings."""
    import pandas as pd
    rows = []
    for k, name in enumerate(CTX_NAMES):
        row = {'context': name}
        for split in ['train', 'val', 'test']:
            counts = np.bincount(data[split]['ctx'][:, k], minlength=len(CTX_LEVELS[name]))
            row[split] = '  '.join(f'{lv}:{c}' for lv, c in zip(CTX_LEVELS[name], counts))
        rows.append(row)
    return pd.DataFrame(rows)
