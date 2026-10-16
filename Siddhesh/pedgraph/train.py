"""Train / evaluate Pedestrian Graph+ on JAAD (PCPA windows). Recipe of the released code: AdamW (wd 1e-3), OneCycleLR
(max lr 5e-3, div 10, final div 1e4), 30 epochs, fp16, class-weighted BCE, optional time-crop. Binary head (one logit) because the
PCPA protocol has no 'irrelevant' class. Model selection by validation AUC; threshold chosen on validation.

    python -m pedgraph.train --samples S.pkl --cache cache_pose --out runs/pg_IV_s42 --image --vel [--time_crop] --seed 42
Run from the `Siddhesh/` directory. Writes config.json, history.csv, metrics.json, best.pt, test_outputs.npz.
"""
import argparse
import csv
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from routed_vit.metrics import compute_metrics, select_threshold

from .data import PedGraphDataset
from .model import PedGraphPlus


@torch.no_grad()
def predict(model, loader, device, amp, image, vel):
    model.eval()
    ps, ys, ix = [], [], []
    for b in loader:
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            lo = model(b['kp'].to(device), b['frame'].to(device) if image else None, b['vel'].to(device) if vel else None)
        ps.append(torch.sigmoid(lo.float()).squeeze(1).cpu().numpy())
        ys.append(b['y'].numpy())
        ix.append(b['idx'].numpy())
    return np.concatenate(ix), np.concatenate(ys), np.concatenate(ps)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--image', action='store_true', help='local-context image branch (paper: I)')
    ap.add_argument('--vel', action='store_true', help='ego-vehicle branch (paper: V)')
    ap.add_argument('--time_crop', action='store_true', help='paper: T')
    ap.add_argument('--stage1_image_gate', action='store_true')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(argv)
    if os.path.exists(os.path.join(a.out, 'metrics.json')):
        print('[skip] finished:', a.out)
        return
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'
    data = pickle.load(open(a.samples, 'rb'))
    kw = dict(use_image=a.image, use_vel=a.vel)
    tr = PedGraphDataset(data['train'], a.cache, train=True, **kw)
    ev = {s: DataLoader(PedGraphDataset(data[s], a.cache, **kw), 64, num_workers=a.workers) for s in ['val', 'test']}
    tl = DataLoader(tr, a.bs, shuffle=True, drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0)
    y_tr = data['train']['label']
    w1, w0 = len(y_tr) / (2 * y_tr.sum()), len(y_tr) / (2 * (1 - y_tr).sum())
    model = PedGraphPlus(frames=a.image, vel=a.vel, stage1_image_gate=a.stage1_image_gate).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, div_factor=10.0, final_div_factor=1e4,
                                                total_steps=a.epochs * len(tl))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    print(f'PedGraph+ params={n_params} image={a.image} vel={a.vel} seed={a.seed} steps/epoch={len(tl)}', flush=True)
    hist, best, t0 = [], -1.0, time.time()
    for ep in range(a.epochs):
        model.train()
        tot = n = 0
        for b in tl:
            y = b['y'].to(dev)
            kp = b['kp'].to(dev)
            if a.time_crop and np.random.randint(10) >= 5:   # keep only the last k frames of the pose sequence (paper: T)
                kp = kp[:, :, -np.random.randint(2, kp.shape[2]):]
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                lo = model(kp, b['frame'].to(dev) if a.image else None, b['vel'].to(dev) if a.vel else None)
            wt = torch.where(y > 0.5, torch.tensor(w1, device=dev), torch.tensor(w0, device=dev))
            loss = (F.binary_cross_entropy_with_logits(lo.float().squeeze(1), y, reduction='none') * wt).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(y)
            n += len(y)
        _, yv, pv = predict(model, ev['val'], dev, amp, a.image, a.vel)
        m = compute_metrics(yv, pv)
        hist.append({'epoch': ep + 1, 'train_loss': tot / n, 'val_auc': m['auc'], 'val_bench_f1': m['bench_f1'], 'min': (time.time() - t0) / 60})
        print(' '.join(f'{k}={v:.4f}' for k, v in hist[-1].items()), flush=True)
        if m['auc'] > best:
            best = m['auc']
            torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
    model.load_state_dict(torch.load(os.path.join(a.out, 'best.pt'), map_location=dev))
    _, yv, pv = predict(model, ev['val'], dev, amp, a.image, a.vel)
    ix, yt, pt = predict(model, ev['test'], dev, amp, a.image, a.vel)
    thr = select_threshold(yv, pv)
    met = {'val': compute_metrics(yv, pv, thr), 'test': compute_metrics(yt, pt, thr), 'n_params': n_params, 'best_val_auc': best}
    np.savez_compressed(os.path.join(a.out, 'test_outputs.npz'), idx=ix, y=yt, p=pt)
    with open(os.path.join(a.out, 'history.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(hist[0]))
        wr.writeheader()
        wr.writerows(hist)
    json.dump(vars(a), open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
    json.dump(met, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    print('TEST', {k: round(v, 4) for k, v in met['test'].items() if k in ('auc', 'bench_auc', 'bench_acc', 'bench_f1')})


if __name__ == '__main__':
    main()
