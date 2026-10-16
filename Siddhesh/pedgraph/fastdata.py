"""GPU-resident data + augmentation + feature construction for the optimised Pedestrian Graph+ pipeline (round 2).

All 4257 JAAD_beh windows (train+val pooled, then test) live on the GPU as tensors, so one epoch over ~1900 windows takes a
fraction of a second and cross-validated hyper-parameter search becomes affordable. Per-window content:
    kp   [T=16, 19, 3]  ViTPose keypoints in pixels (x, y, score)           (cache written by pedgraph.extract)
    box  [16, 4]        annotated pedestrian box (x1, y1, x2, y2) in pixels
    wh   [2]            image width / height
    veh  [16]           JAAD vehicle-action category 0..4 (stopped, slow, fast, decelerating, accelerating)
    img  [3,192,64]     local-context crop of the last observed frame (uint8)
Rows 0..n_pool-1 are the pooled train+val windows (cross-validation), rows n_pool.. the official test windows.
"""
import pickle

import numpy as np
import torch

from routed_vit.cv import pool_trainval

# COCO-17 + neck (17) + pelvis (18)
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 17), (17, 5), (17, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 18), (18, 11),
            (18, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18]   # left <-> right joint permutation (horizontal flip)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PoseData:
    def __init__(self, samples_pkl, cache_dir, device, with_images=True):
        data = pickle.load(open(samples_pkl, 'rb'))
        pool, test = pool_trainval(data), data['test']
        self.pool_rec, self.test_rec = pool, test
        self.n_pool, self.n_test = len(pool['label']), len(test['label'])
        self.n_train = len(data['train']['label'])      # pooled rows [0, n_train) are the official train split, the rest official val
        recs = [pool, test]
        kp, box, wh, veh, y, img = [], [], [], [], [], []
        vids, pids = [], []
        cache = {}

        def video(vid):
            if vid not in cache:
                z = np.load(f'{cache_dir}/{vid}.npz')
                p = list(z['pids'])
                cache[vid] = ({(p[a], int(f)): r for r, (a, f) in enumerate(z['keys'])},
                              {(p[a], int(f)): r for r, (a, f) in enumerate(z['crop_keys'])}, z['kps'], z['crops'])
            return cache[vid]

        for rec in recs:
            for i in range(len(rec['label'])):
                vid, pid = rec['vid'][i], str(rec['pid'][i])
                kix, cix, kps, crops = video(vid)
                frames = [int(f) for f in rec['frames'][i]]
                kp.append(np.stack([kps[kix[(pid, f)]] for f in frames]))
                if with_images:
                    img.append(crops[cix[(pid, frames[-1])]].transpose(2, 0, 1))
                box.append(rec['bbox'][i])
                wh.append(rec['img_wh'][i])
                veh.append(rec['vehicle_act'][i])
                y.append(rec['label'][i])
                vids.append(vid)
                pids.append(pid)
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)
        self.kp, self.box, self.wh = f32(np.stack(kp)), f32(np.stack(box)), f32(np.stack(wh))
        self.veh = torch.from_numpy(np.stack(veh).astype(np.int64)).to(device)
        self.y = f32(np.array(y))
        self.img = torch.from_numpy(np.stack(img)).to(device) if with_images else None
        self.vid, self.pid = np.array(vids), np.array(pids)
        self.groups = np.array([f'{v}/{p}' for v, p in zip(vids, pids)])
        self.device = device
        cy = (self.box[:self.n_pool, :, 1] + self.box[:self.n_pool, :, 3]) / 2
        self.ymin = float(cy.min())                     # GTransPDM: Y_min over the (pooled train+val) samples
        self.pool_rows = np.arange(self.n_pool)
        self.test_rows = self.n_pool + np.arange(self.n_test)
        self.y_np = np.array(y, dtype=np.float32)


# --------------------------------------------------------------------------- features + augmentation
def n_pose_channels(cfg):
    if cfg['arch'] == 'gtranspdm':
        return 3
    c = 1                                                      # keypoint score
    c += 2 * ('abs' in cfg['pose_repr']) + 2 * ('rel' in cfg['pose_repr'])
    if cfg['kp_vel']:
        c += 2
    return c


def n_side_channels(cfg):
    if cfg['arch'] == 'gtranspdm':
        return 9
    s = cfg['side']
    return {'none': 0, 'ego': 5, 'ego_scalar': 2, 'box': 8, 'box+ego': 13}[s]


def _flip_box(box, W):
    return torch.stack([W - box[..., 2], box[..., 1], W - box[..., 0], box[..., 3]], -1)


def build_batch(D, rows, cfg, train, flip=None, gen=None):
    """-> pose [B,C,T,V], side [B,Cs,T] or None, img [B,3,192,64] or None. `flip` (bool [B]) forces horizontal flips (TTA)."""
    if cfg['arch'] == 'gtranspdm':
        from .gtranspdm import build_batch_g
        return build_batch_g(D, rows, cfg, train)
    dev = D.device
    rows_t = torch.as_tensor(rows, device=dev)
    kp, box, wh = D.kp[rows_t].clone(), D.box[rows_t].clone(), D.wh[rows_t]
    B, T, V, _ = kp.shape
    W, H = wh[:, 0].view(B, 1, 1), wh[:, 1].view(B, 1, 1)
    rnd = lambda *s: torch.rand(*s, device=dev, generator=gen)
    rndn = lambda *s: torch.randn(*s, device=dev, generator=gen)
    if flip is None:
        flip = (rnd(B) < cfg['hflip']) if train else torch.zeros(B, dtype=torch.bool, device=dev)
    if flip.any():
        kpf = kp[:, :, LR_SWAP].clone()
        kpf[..., 0] = W - kpf[..., 0]
        kp = torch.where(flip.view(B, 1, 1, 1), kpf, kp)
        box = torch.where(flip.view(B, 1, 1), _flip_box(box, W[..., 0]), box)
    bw = (box[..., 2] - box[..., 0]).clamp(min=2.0)
    bh = (box[..., 3] - box[..., 1]).clamp(min=2.0)
    cx, cy = (box[..., 0] + box[..., 2]) / 2, (box[..., 1] + box[..., 3]) / 2
    if train:
        if cfg['gscale'] > 0:   # one random zoom / shift per window, same for every frame (keeps the motion pattern)
            g = cfg['gscale']
            s = 1 + g * rndn(B, 1)
            tx, ty = g * rndn(B, 1) * bw[:, -1:], g * rndn(B, 1) * bh[:, -1:]
            cx0, cy0 = cx[:, -1:], cy[:, -1:]
            kp[..., 0] = cx0[..., None] + s[..., None] * (kp[..., 0] - cx0[..., None]) + tx[..., None]
            kp[..., 1] = cy0[..., None] + s[..., None] * (kp[..., 1] - cy0[..., None]) + ty[..., None]
            cx, cy = cx0 + s * (cx - cx0) + tx, cy0 + s * (cy - cy0) + ty
            bw, bh = bw * s, bh * s
        if cfg['jitter'] > 0:   # per-keypoint noise, proportional to the person's size
            kp[..., 0] += rndn(B, T, V) * cfg['jitter'] * bw[..., None]
            kp[..., 1] += rndn(B, T, V) * cfg['jitter'] * bh[..., None]
        if cfg['joint_drop'] > 0:   # a missed detection: joint moves to the box centre and its score becomes 0
            drop = rnd(B, T, V) < cfg['joint_drop']
            kp[..., 0] = torch.where(drop, cx[..., None].expand(-1, -1, V), kp[..., 0])
            kp[..., 1] = torch.where(drop, cy[..., None].expand(-1, -1, V), kp[..., 1])
            kp[..., 2] = kp[..., 2] * (~drop)
    ch = []
    rel = None
    if 'abs' in cfg['pose_repr']:
        ch += [(kp[..., 0] / W).clamp(0, 1), (kp[..., 1] / H).clamp(0, 1)]
    if 'rel' in cfg['pose_repr'] or cfg['kp_vel']:
        rel = [(kp[..., 0] - cx[..., None]) / bw[..., None], (kp[..., 1] - cy[..., None]) / bh[..., None]]
        if 'rel' in cfg['pose_repr']:
            ch += rel
    if cfg['kp_vel']:
        base = rel if rel is not None else ch[:2]
        for c in base:
            d = torch.zeros_like(c)
            d[:, 1:] = c[:, 1:] - c[:, :-1]
            ch.append(d * 5)
    ch.append(kp[..., 2])
    pose = torch.stack(ch, 1)                      # [B,C,T,V]
    side = None
    if cfg['side'] != 'none':
        parts = []
        if 'box' in cfg['side']:
            ab = torch.stack([cx / W[..., 0], cy / H[..., 0], bw / W[..., 0], bh / H[..., 0]], -1)          # [B,T,4]
            parts += [ab, (ab - ab[:, -1:]) * 5]
        if 'ego' in cfg['side']:
            v = D.veh[rows_t]
            parts.append(torch.nn.functional.one_hot(v, 5).float() if cfg['side'] != 'ego_scalar' else v.float()[..., None].expand(-1, -1, 2))
        side = torch.cat(parts, -1).permute(0, 2, 1).contiguous()    # [B,Cs,T]
    img = None
    if cfg['use_image']:
        x = D.img[rows_t].float() / 255.0
        x = torch.where(flip.view(B, 1, 1, 1), x.flip(-1), x)
        if train and cfg['img_aug'] > 0:
            a = cfg['img_aug']
            x = x * (1 + a * 0.3 * (rnd(B, 1, 1, 1) * 2 - 1))                                            # contrast-ish gain
            x = x + a * 0.15 * (rnd(B, 1, 1, 1) * 2 - 1)                                                 # brightness
            x = x + a * 0.03 * rndn(*x.shape)
            e = rnd(B) < 0.5 * a                                                                         # random erasing
            hh, ww = x.shape[2], x.shape[3]
            ys, xs = (rnd(B) * hh * 0.6).long(), (rnd(B) * ww * 0.6).long()
            gy, gx = torch.arange(hh, device=dev).view(1, hh, 1), torch.arange(ww, device=dev).view(1, 1, ww)
            m = e.view(B, 1, 1) & (gy >= ys.view(B, 1, 1)) & (gy < ys.view(B, 1, 1) + hh // 4) & (gx >= xs.view(B, 1, 1)) & (gx < xs.view(B, 1, 1) + ww // 3)
            x = torch.where(m.unsqueeze(1), torch.full_like(x, 0.5), x)
            x = x.clamp(0, 1)
        img = (x - MEAN.to(dev)) / STD.to(dev)
        if cfg['img_down'] > 1:   # 192x64 -> 96x32: 4x cheaper image branch
            img = torch.nn.functional.avg_pool2d(img, cfg['img_down'])
    return pose, side, img
