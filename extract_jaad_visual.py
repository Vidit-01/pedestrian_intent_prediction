"""Precompute frozen Swin-V2-B feature maps (8x8x1024, fp16) for the visual modalities used by ACIT
(and, pooled, by the visual GRU ablations). Reads frames straight from the mp4 clips (no PNG extraction).

Modalities (all resized to 256x256 before Swin, as in ACIT Sec. III-B):
  global_seg  : SegFormer (Cityscapes, 19 classes) argmax, coloured with the Cityscapes palette
  global_flow : RAFT-large flow (frame t-1 -> t; for t=0: 0 -> 1), coloured with torchvision flow_to_image
  local_rgb   : RGB crop of the pedestrian box (original box size, no enlargement)
  local_flow  : the same box cropped from the coloured global flow image
  global_rgb  : full RGB frame (not used by ACIT; scene input for Baseline A ablation 5)

Output layout (one file per video, resumable):
  <out>/<modality>/<vid>.npy   float16 [n, 64, 1024]
  <out>/index/<vid>.json       {"global_frames": [f, ...], "local_keys": [[pid, f], ...]}  (row order of the npy files)
  <out>/meta.json              models / resolutions used

Two GPUs: videos are sharded by rank, no communication needed:
    torchrun --nproc_per_node=2 extract_jaad_visual.py --jaad_root ... --samples a.pkl b.pkl --out /tmp/jaad_feats
"""
import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ALL_MODALITIES = ['local_rgb', 'local_flow', 'global_seg', 'global_flow', 'global_rgb']


def collect_needs(sample_paths):
    """vid -> {'frames': sorted global frames, 'local': {frame: [(pid, bbox), ...]}} over all splits and files."""
    needs = defaultdict(lambda: {'frames': set(), 'local': defaultdict(dict)})
    for p in sample_paths:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        for s in ['train', 'val', 'test']:
            r = data[s]
            for vid, pid, frames, boxes in zip(r['vid'], r['pid'], r['frames'], r['bbox']):
                for fr, b in zip(frames, boxes):
                    needs[vid]['frames'].add(int(fr))
                    needs[vid]['local'][int(fr)][pid] = b
    return {v: {'frames': sorted(n['frames']), 'local': {f: sorted(d.items()) for f, d in n['local'].items()}}
            for v, n in needs.items()}


def crop(img, box, sx=1.0, sy=1.0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    x1, y1 = int(np.clip(np.floor(x1), 0, w - 1)), int(np.clip(np.floor(y1), 0, h - 1))
    x2, y2 = int(np.clip(np.ceil(x2), x1 + 1, w)), int(np.clip(np.ceil(y2), y1 + 1, h))
    return img[y1:y2, x1:x2]


def to256(img, nearest=False):
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


class Extractor:
    def __init__(self, args, device):
        import timm
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        from transformers import SegformerForSemanticSegmentation

        self.args, self.device = args, device
        self.swin = timm.create_model(args.swin, pretrained=True, num_classes=0).eval().to(device).half()
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(device)
        self.seg = SegformerForSemanticSegmentation.from_pretrained(args.segformer).eval().to(device).half()
        self.mean, self.std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    @torch.no_grad()
    def swin_encode(self, imgs):
        """list of 256x256x3 uint8 RGB -> float16 [n, 64, 1024]"""
        outs = []
        for i in range(0, len(imgs), self.args.swin_batch):
            x = torch.from_numpy(np.stack(imgs[i:i + self.args.swin_batch])).to(self.device)
            x = ((x.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std).half()
            f = self.swin.forward_features(x)  # timm swinv2: [n, 8, 8, 1024] (NHWC, after final norm)
            outs.append(f.reshape(f.shape[0], -1, f.shape[-1]).cpu().numpy().astype(np.float16))
        return np.concatenate(outs)

    @torch.no_grad()
    def flow_images(self, prev, cur):
        """lists of full-res RGB uint8 -> coloured flow uint8 [n, fh, fw, 3] at --flow_hw"""
        from torchvision.utils import flow_to_image
        fh, fw = self.args.flow_hw

        def prep(lst):
            x = torch.from_numpy(np.stack([cv2.resize(im, (fw, fh)) for im in lst])).to(self.device)
            return x.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # RAFT expects [-1, 1]

        a, b = prep(prev), prep(cur)
        out = []
        for i in range(0, len(a), self.args.raft_batch):
            flow = self.raft(a[i:i + self.args.raft_batch], b[i:i + self.args.raft_batch])[-1]
            out.append(flow_to_image(flow).permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def seg_images(self, rgbs):
        """list of full-res RGB uint8 -> Cityscapes-coloured label maps uint8 [n, sh/4, sw/4, 3]"""
        sh, sw = self.args.seg_hw
        x = torch.from_numpy(np.stack([cv2.resize(im, (sw, sh)) for im in rgbs])).to(self.device)
        x = ((x.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std).half()
        labels = []
        for i in range(0, len(x), self.args.seg_batch):
            labels.append(self.seg(pixel_values=x[i:i + self.args.seg_batch]).logits.argmax(1).cpu().numpy())
        return CITYSCAPES_PALETTE[np.concatenate(labels)]

    def process_chunk(self, chunk, local, mods, store):
        """chunk: list of (frame, rgb, flow_prev, flow_cur). Appends features + keys to store."""
        rgbs = [c[1] for c in chunk]
        H, W = rgbs[0].shape[:2]
        need_flow = 'global_flow' in mods or 'local_flow' in mods
        flows = self.flow_images([c[2] for c in chunk], [c[3] for c in chunk]) if need_flow else None
        segs = self.seg_images(rgbs) if 'global_seg' in mods else None
        imgs = {m: [] for m in mods}
        for j, (f, rgb, _, _) in enumerate(chunk):
            store['global_frames'].append(f)
            if 'global_rgb' in mods:
                imgs['global_rgb'].append(to256(rgb))
            if 'global_seg' in mods:
                imgs['global_seg'].append(to256(segs[j], nearest=True))
            if 'global_flow' in mods:
                imgs['global_flow'].append(to256(flows[j]))
            for pid, box in local.get(f, []):
                store['local_keys'].append([pid, f])
                if 'local_rgb' in mods:
                    imgs['local_rgb'].append(to256(crop(rgb, box)))
                if 'local_flow' in mods:
                    fl = flows[j]
                    imgs['local_flow'].append(to256(crop(fl, box, fl.shape[1] / W, fl.shape[0] / H)))
        for m in mods:
            if imgs[m]:
                store[m].append(self.swin_encode(imgs[m]))
        return imgs

    def run_video(self, vid, need, mods):
        args = self.args
        needed = set(need['frames'])
        last = max(max(needed), 1)
        cap = cv2.VideoCapture(os.path.join(args.jaad_root, 'JAAD_clips', vid + '.mp4'))
        store = {'global_frames': [], 'local_keys': [], **{m: [] for m in mods}}
        chunk, prev, vis_left = [], None, args.vis
        for i in range(last + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'{vid}: could not read frame {i} (need up to {last})')
            cur = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if i in needed and i > 0:
                chunk.append((i, cur, prev, cur))
            if i == 1 and 0 in needed:
                chunk.append((0, prev, prev, cur))
            if len(chunk) >= args.chunk or (i == last and chunk):
                imgs = self.process_chunk(chunk, need['local'], mods, store)
                if vis_left > 0:
                    vis_left -= self.save_vis(vid, chunk, need['local'], imgs, vis_left)
                chunk = []
            prev = cur
        cap.release()
        return store

    def save_vis(self, vid, chunk, local, imgs, n):
        """Side-by-side check of what Swin sees: local_rgb | local_flow | global_rgb | global_seg | global_flow."""
        os.makedirs(os.path.join(self.args.out, 'vis'), exist_ok=True)
        k, saved = 0, 0
        for j, (f, *_ ) in enumerate(chunk):
            for pid, _ in local.get(f, []):
                if saved < n:
                    row = [imgs[m][k] for m in ['local_rgb', 'local_flow'] if imgs.get(m)]
                    row += [imgs[m][j] for m in ['global_rgb', 'global_seg', 'global_flow'] if imgs.get(m)]
                    cv2.imwrite(os.path.join(self.args.out, 'vis', f'{vid}_{f:05d}_{pid}.jpg'),
                                cv2.cvtColor(np.concatenate(row, 1), cv2.COLOR_RGB2BGR))
                    saved += 1
                k += 1
        return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jaad_root', required=True)
    ap.add_argument('--samples', nargs='+', required=True, help='sample pkl(s) from build_jaad_samples.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--modalities', nargs='+', default=ALL_MODALITIES, choices=ALL_MODALITIES)
    ap.add_argument('--swin', default='swinv2_base_window8_256.ms_in1k')
    ap.add_argument('--segformer', default='nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    ap.add_argument('--flow_hw', type=int, nargs=2, default=[544, 960], help='RAFT input size (multiple of 8)')
    ap.add_argument('--seg_hw', type=int, nargs=2, default=[544, 960], help='SegFormer input size')
    ap.add_argument('--chunk', type=int, default=16, help='frames decoded per GPU step')
    ap.add_argument('--raft_batch', type=int, default=8)
    ap.add_argument('--seg_batch', type=int, default=8)
    ap.add_argument('--swin_batch', type=int, default=64)
    ap.add_argument('--vis', type=int, default=0, help='save N side-by-side check images per video (rank 0 only)')
    ap.add_argument('--max_videos', type=int, default=0, help='debug: only the first N videos')
    args = ap.parse_args()

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if rank != 0:
        args.vis = 0
    for d in args.modalities + ['index']:
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    if rank == 0:
        with open(os.path.join(args.out, 'meta.json'), 'w') as f:
            json.dump(vars(args), f, indent=1)

    needs = collect_needs(args.samples)
    vids = sorted(needs)[:args.max_videos or None][rank::world]
    ex = Extractor(args, device)
    t0 = time.time()
    for n, vid in enumerate(vids):
        idx_path = os.path.join(args.out, 'index', vid + '.json')
        if os.path.exists(idx_path) and all(os.path.exists(os.path.join(args.out, m, vid + '.npy')) for m in args.modalities):
            continue
        store = ex.run_video(vid, needs[vid], args.modalities)
        for m in args.modalities:
            np.save(os.path.join(args.out, m, vid + '.npy'), np.concatenate(store[m]))
        with open(idx_path, 'w') as f:  # written last: marks the video as complete
            json.dump({'global_frames': store['global_frames'], 'local_keys': store['local_keys']}, f)
        print(f'[rank {rank}] {n + 1}/{len(vids)} {vid}: {len(store["global_frames"])} frames, '
              f'{len(store["local_keys"])} crops, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
