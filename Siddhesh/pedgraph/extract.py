"""Pose + local-context cache for Pedestrian Graph+ on the PCPA windows of build_jaad_samples.py.

JAAD ships no skeletons and the authors' pre-extracted keypoints are not reproducible from the released code, so poses are
estimated top-down with ViTPose (HF `usyd-community/vitpose-base-simple`) on the *annotated* pedestrian boxes:
17 COCO joints + neck (mid-shoulders) + pelvis (mid-hips) = 19 nodes, channels (x_px, y_px, score).
Per video one npz:  keys [n,2] (pid-index, frame), pids [list], kps [n,19,3] float32,
                    crop_keys [m,2], crops [m,192,64,3] uint8  (box crop of the LAST observed frame of each window, tall 3:1 like the paper)

    python -m pedgraph.extract --samples S.pkl --clips JAAD/JAAD_clips --out cache_pose [--vitpose usyd-community/vitpose-base-simple]
Resumable (finished videos are skipped).
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch

CROP_HW = (192, 64)


def to19(kps, sc):
    """COCO-17 [17,2],[17] -> 19 nodes [19,3] = (x, y, score) with neck and pelvis appended."""
    k = np.concatenate([kps, sc[:, None]], 1)
    neck = np.array([*(kps[5] + kps[6]) / 2, min(sc[5], sc[6])])
    pelvis = np.array([*(kps[11] + kps[12]) / 2, min(sc[11], sc[12])])
    return np.concatenate([k, neck[None], pelvis[None]], 0).astype(np.float32)


def box_crop(img, box, margin=0.1, hw=CROP_HW):
    x1, y1, x2, y2 = [float(b) for b in box]
    w, h = x2 - x1, y2 - y1
    x1, x2, y1, y2 = x1 - margin * w, x2 + margin * w, y1 - margin * h, y2 + margin * h
    H, W = img.shape[:2]
    xa, ya, xb, yb = int(max(0, np.floor(x1))), int(max(0, np.floor(y1))), int(min(W, np.ceil(x2))), int(min(H, np.ceil(y2)))
    if xb <= xa or yb <= ya:
        return np.zeros((*hw, 3), np.uint8)
    return cv2.resize(img[ya:yb, xa:xb], (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


class PoseEstimator:
    def __init__(self, name='usyd-community/vitpose-base-simple', device='cuda', fp16=True):
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.device = torch.device(device)
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = VitPoseForPoseEstimation.from_pretrained(name).to(self.device).eval()
        self.fp16 = fp16 and self.device.type == 'cuda'
        if self.fp16:
            self.model.half()

    @torch.no_grad()
    def __call__(self, rgb, boxes_xyxy):
        """rgb uint8 [H,W,3]; boxes [n,4] (x1,y1,x2,y2) -> kps [n,19,3]"""
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
        coco = boxes.copy()
        coco[:, 2] = np.maximum(boxes[:, 2] - boxes[:, 0], 2)   # xywh
        coco[:, 3] = np.maximum(boxes[:, 3] - boxes[:, 1], 2)
        inp = self.proc(images=rgb, boxes=[coco], return_tensors='pt').to(self.device)
        if self.fp16:
            inp['pixel_values'] = inp['pixel_values'].half()
        out = self.model(**inp)
        out.heatmaps = out.heatmaps.float()
        res = self.proc.post_process_pose_estimation(out, boxes=[coco])[0]
        return np.stack([to19(r['keypoints'].float().cpu().numpy(), r['scores'].float().cpu().numpy()) for r in res])


def needed_per_video(data):
    """vid -> {'items': {(pid, frame): bbox}, 'last': {(pid, frame)}} over all 16 frames of every window."""
    need = {}
    for split in ['train', 'val', 'test']:
        rec = data[split]
        for i in range(len(rec['label'])):
            d = need.setdefault(rec['vid'][i], {'items': {}, 'last': set()})
            for t in range(rec['frames'].shape[1]):
                d['items'][(rec['pid'][i], int(rec['frames'][i][t]))] = rec['bbox'][i][t]
            d['last'].add((rec['pid'][i], int(rec['frames'][i][-1])))
    return need


def extract(data, clips_dir, out_dir, pose_name='usyd-community/vitpose-base-simple', device='cuda', verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    need = needed_per_video(data)
    est = None
    for n, (vid, d) in enumerate(sorted(need.items())):
        path = os.path.join(out_dir, vid + '.npz')
        if os.path.exists(path):
            continue
        if est is None:
            est = PoseEstimator(pose_name, device)
        by_frame = {}
        for (pid, fr), box in d['items'].items():
            by_frame.setdefault(fr, []).append((pid, box))
        pids = sorted({p for p, _ in d['items']})
        pid_ix = {p: i for i, p in enumerate(pids)}
        cap = cv2.VideoCapture(os.path.join(clips_dir, vid + '.mp4'))
        keys, kps, ckeys, crops = [], [], [], []
        for fr in range(max(by_frame) + 1):
            if not cap.grab():
                raise RuntimeError(f'{vid}: cannot read frame {fr}')
            if fr not in by_frame:
                continue
            _, bgr = cap.retrieve()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            items = by_frame[fr]
            k = est(rgb, np.stack([b for _, b in items]))
            for (pid, box), kk in zip(items, k):
                keys.append((pid_ix[pid], fr))
                kps.append(kk)
                if (pid, fr) in d['last']:
                    ckeys.append((pid_ix[pid], fr))
                    crops.append(box_crop(rgb, box))
        cap.release()
        np.savez_compressed(path + '.tmp.npz', keys=np.array(keys), pids=np.array(pids), kps=np.stack(kps),
                            crop_keys=np.array(ckeys), crops=np.stack(crops))
        os.replace(path + '.tmp.npz', path)
        if verbose:
            print(f'[{n + 1}/{len(need)}] {vid}: {len(keys)} poses, {len(crops)} crops', flush=True)
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--clips', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--vitpose', default='usyd-community/vitpose-base-simple')
    a = ap.parse_args()
    extract(pickle.load(open(a.samples, 'rb')), a.clips, a.out, a.vitpose)
