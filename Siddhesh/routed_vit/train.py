"""Train + evaluate one (variant, seed, fold) job of the PedViViT-style backbone on one GPU.   (idea.md sections 6 and 10)

    CUDA_VISIBLE_DEVICES=0 python -m routed_vit.train --variant soft --seed 42 --fold 0 --samples S.pkl --crops C --out runs/soft_s42_f0

Protocol (round 2). Round 1 selected the best epoch on the 22-pedestrian validation split and early-stopped; every run peaked at
epoch 1 and then memorised the training set (train loss 1e-5), i.e. the "trained" model was a one-epoch checkpoint. Now:
  * train+val are pooled and split into K video-grouped folds (cv.py); `--fold i` trains on K-1 folds and validates on the held-out one;
    `--fold -1` reproduces the official train/val split. The official test split is only predicted, never used for selection.
  * FIXED budget, no early stopping, no best-epoch selection: warm-up -> cosine to ~0 over `--epochs`, and the model that is
    evaluated is the EMA of the weights at the END of the schedule. The same rule applies to every variant.
  * regularisation against memorisation: lower LR with layer-wise decay (`--llrd`), optional frozen prefix (`--freeze_blocks`),
    stochastic depth, label smoothing, EMA, clip-consistent affine jitter, box noise, box-branch modality dropout.
Identical for every variant: pretrained init, folds, augmentation, optimizer, schedule, step count, seeds, selection rule (none).
Writes to --out: config.json, best.pt (final EMA weights, fp16), history.csv, metrics.json, test_outputs.npz, val_outputs.npz
(+ test_curve.npz with --track_test: diagnostic only, never used for selection).
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cv import make_folds, pool_trainval
from .data import CropDataset, load_samples, prep
from .metrics import compute_metrics
from .model import RoutedViT, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def entropy(p, axis=-1):
    return -(p * np.log(p + 1e-12)).sum(axis)


@torch.no_grad()
def predict(model, loader, device, amp=True, tau=1.0):
    """Returns dict of arrays: idx, y, logit, p and, for routed variants, z [n,d], gates/pi [n,Lr,H], mask [n,Lr,H]."""
    model.eval()
    out = {k: [] for k in ['idx', 'y', 'logit', 'z', 'gates', 'pi', 'mask']}
    for b in loader:
        x = None if model.box_only else prep(b['x'].to(device, non_blocking=True))
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            logit, info = model(x, box=b['box'].to(device), tau=tau)
        out['idx'].append(b['idx'].numpy())
        out['y'].append(b['y'].numpy())
        out['logit'].append(logit.float().cpu().numpy())
        if info['gates']:
            out['z'].append(info['z'].cpu().numpy().astype(np.float16))
            out['gates'].append(torch.stack(info['gates'], 1).cpu().numpy())
            if info['pi']:
                out['pi'].append(torch.stack(info['pi'], 1).cpu().numpy())
            if info['mask'] and info['mask'][0] is not None:
                out['mask'].append(torch.stack(info['mask'], 1).cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items() if v}
    res['p'] = 1 / (1 + np.exp(-res['logit']))
    return res


class EMA:
    """Exponential moving average of the parameters (fp32 shadow). The model has no buffers, so parameters are the whole state."""

    def __init__(self, model, decay):
        self.decay = decay
        self.params = [p for p in model.parameters()]
        self.shadow = [p.detach().clone().float() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self, step):
        d = min(self.decay, (1 + step) / (10 + step))    # warm start: do not average in the random-ish first steps
        torch._foreach_mul_(self.shadow, d)
        torch._foreach_add_(self.shadow, [p.detach().float() for p in self.params], alpha=1 - d)

    @torch.no_grad()
    def apply(self):
        self.backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.copy_(s.to(p.dtype))

    @torch.no_grad()
    def restore(self):
        for p, b in zip(self.params, self.backup):
            p.copy_(b)
        self.backup = None


def layer_id(name, depth):
    """Depth index for layer-wise LR decay: embeddings 0, block i -> i+1, final norm depth+1."""
    if name.startswith(('patch_embed', 'cls_token', 'pos_embed')):
        return 0
    if name.startswith('blocks.'):
        return int(name.split('.')[1]) + 1
    return depth + 1


def make_optimizer(model, args):
    depth = len(model.blocks)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(('routers', 'film', 'z_norm')):
            g, lr = 'router', args.lr * args.router_lr_mult
        elif n.startswith(('temporal', 't_cls', 't_pos', 'nv', 'fuse')):
            g, lr = 'head', args.head_lr
        else:
            g, lr = 'backbone', args.lr * args.llrd ** (depth + 1 - layer_id(n, depth))
        decay = p.ndim > 1 and not n.endswith(('pos_embed', 't_pos', 'nv_pos', 'cls_token', 't_cls'))
        groups.setdefault((g, round(lr, 12), decay), []).append(p)
    pg = [{'params': ps, 'weight_decay': args.wd if decay else 0.0, 'group': g, 'base_lr': lr, 'lr': lr}
          for (g, lr, decay), ps in groups.items()]
    return torch.optim.AdamW(pg, betas=(0.9, 0.999))


def lr_scale(step, total, warmup):
    w = max(1, int(warmup * total))
    if step < w:
        return (step + 1) / w
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['dense', 'soft', 'sparse', 'film'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fold', type=int, default=0, help='CV fold held out for validation; -1 = official train/val split')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--fold_seed', type=int, default=0, help='fold assignment seed (same for every variant / training seed)')
    ap.add_argument('--samples', required=True)
    ap.add_argument('--crops', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--micro_bs', type=int, default=16, help='clips per forward (x8 frames = images)')
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-5, help='backbone LR at the top block (lower blocks decay by --llrd per block)')
    ap.add_argument('--llrd', type=float, default=0.8, help='layer-wise LR decay of the spatial ViT')
    ap.add_argument('--head_lr', type=float, default=5e-4)
    ap.add_argument('--router_lr_mult', type=float, default=10.0)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=float, default=0.1)
    ap.add_argument('--router_freeze', type=float, default=0.05, help='fraction of steps with the router frozen')
    ap.add_argument('--L0', type=int, default=6)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--lam_bal', type=float, default=1e-2)
    ap.add_argument('--lam_z', type=float, default=1e-3)
    ap.add_argument('--tau0', type=float, default=2.0)
    ap.add_argument('--tau_frac', type=float, default=0.2, help='tau anneals tau0 -> 1 over this fraction of steps')
    ap.add_argument('--sparse_start', type=float, default=0.3,
                    help='B-sparse: first this fraction of steps runs soft (Prop. 1 nesting), then switches to top-k')
    ap.add_argument('--freeze_blocks', type=int, default=0, help='freeze patch embedding + first N ViT blocks (N <= L0)')
    ap.add_argument('--drop_path', type=float, default=0.1, help='stochastic depth, linear to this value at the last block')
    ap.add_argument('--label_smooth', type=float, default=0.05)
    ap.add_argument('--ema', type=float, default=0.995, help='EMA decay of the weights (0 disables); the EMA model is evaluated')
    ap.add_argument('--geo_aug', type=float, default=1.0, help='strength of the clip-consistent affine jitter (0 disables)')
    ap.add_argument('--box_noise', type=float, default=0.02, help='gaussian noise on the delta-encoded boxes (train only)')
    ap.add_argument('--box_drop', type=float, default=0.15, help='probability of zeroing the box-branch feature (train only)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--no_box', action='store_true', help='drop the non-visual (bbox) branch')
    ap.add_argument('--box_only', action='store_true', help='control: bbox branch only, no pixels (dense variant)')
    ap.add_argument('--track_test', action='store_true', help='diagnostic: also store test predictions after every epoch')
    ap.add_argument('--max_train', type=int, default=0, help='debug: use only this many train clips')
    args = ap.parse_args(argv)

    if os.path.exists(os.path.join(args.out, 'metrics.json')):
        print(f'[skip] {args.out} already finished')
        return
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and not args.no_amp
    set_seed(args.seed)

    data = load_samples(args.samples)
    pool = pool_trainval(data)
    if args.fold < 0:
        n_tr = len(data['train']['label'])
        tr_rows, va_rows = np.arange(n_tr), np.arange(n_tr, len(pool['label']))
    else:
        tr_rows, va_rows = make_folds(pool, args.n_folds, args.fold_seed)[args.fold]
    if args.max_train:
        tr_rows = tr_rows[:args.max_train]
    img = not args.box_only
    train_ds = CropDataset(pool, args.crops, indices=tr_rows, images=img)
    val_ds = CropDataset(pool, args.crops, indices=va_rows, images=img)
    test_ds = CropDataset(data['test'], args.crops, images=img)
    g = torch.Generator()
    g.manual_seed(args.seed * 100 + max(args.fold, 0))
    tl = DataLoader(train_ds, args.micro_bs, shuffle=True, drop_last=True, num_workers=args.workers, generator=g,
                    pin_memory=True, persistent_workers=args.workers > 0)
    ev = {'val': DataLoader(val_ds, 32, shuffle=False, num_workers=args.workers, pin_memory=True),
          'test': DataLoader(test_ds, 32, shuffle=False, num_workers=args.workers, pin_memory=True)}

    y_tr = pool['label'][tr_rows]
    w = {1: len(y_tr) / (2 * max(1, y_tr.sum())), 0: len(y_tr) / (2 * max(1, (1 - y_tr).sum()))}   # inverse frequency

    set_seed(args.seed)   # identical shared-weight init for every variant
    model = RoutedViT(args.variant, L0=args.L0, k=args.k, use_box=not args.no_box, drop_path=args.drop_path,
                      freeze_blocks=args.freeze_blocks, box_only=args.box_only, box_drop=args.box_drop)
    if not args.no_pretrained and not args.box_only:
        model.load_pretrained()
    model.to(device)
    n_params, n_router = count_params(model)
    opt = make_optimizer(model, args)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    steps_per_epoch = len(tl) // args.accum
    total = args.epochs * steps_per_epoch
    routed = args.variant in ('soft', 'sparse')
    print(f'{args.variant} seed={args.seed} fold={args.fold} train={len(tr_rows)} val={len(va_rows)} params={n_params / 1e6:.2f}M '
          f'(router/film {n_router / 1e3:.1f}k) steps={total} clips/step={args.micro_bs * args.accum}', flush=True)

    def evaluate(split):
        if ema:
            ema.apply()
        try:
            return predict(model, ev[split], device, amp)
        finally:
            if ema:
                ema.restore()

    hist, step, test_curve = [], 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = {'loss': 0.0, 'task': 0.0, 'n': 0}
        it = iter(tl)
        for _ in range(steps_per_epoch):
            frac = step / total
            sc = lr_scale(step, total, args.warmup)
            for pg in opt.param_groups:
                frozen = pg['group'] == 'router' and frac < args.router_freeze
                pg['lr'] = 0.0 if frozen else pg['base_lr'] * sc
            tau = max(1.0, args.tau0 - (args.tau0 - 1.0) * min(1.0, frac / max(args.tau_frac, 1e-8)))
            sparse_on = frac >= args.sparse_start
            opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                b = next(it)
                x = None if args.box_only else prep(b['x'].to(device, non_blocking=True), train=True, geo=args.geo_aug)
                box = b['box'].to(device)
                if args.box_noise > 0:
                    box = box + args.box_noise * torch.randn_like(box)
                y = b['y'].to(device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    logit, info = model(x, box=box, tau=tau, sparse_on=sparse_on)
                sw = torch.where(y > 0.5, torch.tensor(w[1], device=device), torch.tensor(w[0], device=device))
                ys = y * (1 - args.label_smooth) + 0.5 * args.label_smooth
                task = (F.binary_cross_entropy_with_logits(logit.float(), ys, reduction='none') * sw).mean()
                loss = task + (args.lam_bal * info['bal'] + args.lam_z * info['zloss'] if routed else 0.0)
                scaler.scale(loss / args.accum).backward()
                run['loss'] += float(loss.detach()) * len(y)
                run['task'] += float(task.detach()) * len(y)
                run['n'] += len(y)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if ema:
                ema.update(step)
        vo = evaluate('val')
        m = compute_metrics(vo['y'], vo['p'])
        S = float('nan')
        if 'pi' in vo:
            pi = vo['pi']
            S = float(np.mean(entropy(pi.mean(0)) - entropy(pi).mean(0)))
        row = {'epoch': ep + 1, 'train_loss': run['loss'] / run['n'], 'train_task': run['task'] / run['n'],
               'val_auc': m['auc'], 'val_bench_auc': m['bench_auc'], 'val_f1': m['bench_f1'], 'route_S': S,
               'time_min': (time.time() - t0) / 60}
        if args.track_test:
            to_ep = evaluate('test')
            test_curve.append(to_ep['p'])
            row['test_auc_diag'] = compute_metrics(to_ep['y'], to_ep['p'])['auc']
        hist.append(row)
        print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)

    # final model = EMA weights at the end of the fixed schedule (no checkpoint selection)
    if ema:
        ema.apply()
    torch.save({k: v.half() if v.dtype == torch.float32 else v for k, v in model.state_dict().items()},
               os.path.join(args.out, 'best.pt'))
    if ema:
        ema.restore()
    vo, to = evaluate('val'), evaluate('test')
    metrics = {'val': compute_metrics(vo['y'], vo['p']), 'test': compute_metrics(to['y'], to['p']),
               'n_params': n_params, 'n_router_params': n_router, 'final_val_auc': hist[-1]['val_auc'],
               'epochs_run': len(hist), 'minutes': (time.time() - t0) / 60}
    np.savez_compressed(os.path.join(args.out, 'test_outputs.npz'), **to)
    np.savez_compressed(os.path.join(args.out, 'val_outputs.npz'), **vo)
    if test_curve:
        np.savez_compressed(os.path.join(args.out, 'test_curve.npz'), p=np.stack(test_curve), y=to['y'])
    with open(os.path.join(args.out, 'history.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(hist[0]))
        wr.writeheader()
        wr.writerows(hist)
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:   # written last: marks the run as finished
        json.dump(metrics, f, indent=1)
    print('DONE', {k: round(v, 4) for k, v in metrics['val'].items() if k in ('auc',)},
          'test', {k: round(v, 4) for k, v in metrics['test'].items() if k in ('auc', 'bench_acc', 'bench_f1')})


if __name__ == '__main__':
    main()
