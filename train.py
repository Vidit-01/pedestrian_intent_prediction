"""Train + evaluate any model on the JAAD samples. Single GPU, or DDP across both T4s with torchrun.

    # one GPU (small models; run different seeds on the two GPUs in parallel instead)
    CUDA_VISIBLE_DEVICES=0 python train.py --model gru --samples .../jaad_beh.pkl --kin_feats box --out runs/gru_box_s0
    # two GPUs (ACIT)
    torchrun --nproc_per_node=2 train.py --model acit --samples .../jaad_beh.pkl --feats /tmp/jaad_feats --out runs/acit_s0

T4 has no bf16 -> fp16 autocast + GradScaler. Model selection: best validation AUC (early stopping on it).
The test split is predicted once, with the best checkpoint, after training. Writes to --out:
  config.json, best.pt, history.csv, metrics.json (val + test), val_predictions.csv, test_predictions.csv
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from datasets import JAADDataset, input_dims, kin_stats_from, load_samples
from evaluate import compute_metrics, select_threshold


def build_model(args, dims):
    if args.model == 'gru':
        from models.gru import GRUBaseline
        return GRUBaseline(dims, d=args.d, hidden=args.hidden, num_layers=args.layers, dropout=args.dropout)
    if args.model == 'acit':
        from models.acit import ACIT
        return ACIT(dropout=args.dropout, l2=args.l2_head)
    raise ValueError(args.model)


def model_inputs(args):
    if args.model == 'gru':
        return (['kin'] if args.kin_feats else []) + [f'{m}:pool' for m in args.visual]
    if args.model == 'acit':
        from models.acit import ACIT
        return ACIT.inputs
    raise ValueError(args.model)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ps, ys = [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = model(batch)
        ps.append(torch.sigmoid(logits.float()).cpu())
        ys.append(batch['label'].cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def latency_ms(model, batch, device, amp, reps=50):
    model.eval()
    batch = {k: v[:1].to(device) for k, v in batch.items()}
    for _ in range(5):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            model(batch)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            model(batch)
    torch.cuda.synchronize()
    return (time.time() - t) / reps * 1000


def save_predictions(path, rec, y, p):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['vid', 'pid', 'first_frame', 'tte', 'label', 'prob'])
        for i in range(len(y)):
            w.writerow([rec['vid'][i], rec['pid'][i], rec['frames'][i][0], rec['tte'][i], int(y[i]), f'{p[i]:.6f}'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gru', 'acit'], required=True)
    ap.add_argument('--samples', required=True, help='pkl from build_jaad_samples.py')
    ap.add_argument('--feats', default=None, help='feature root from extract_jaad_visual.py (visual inputs)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--kin_feats', nargs='*', default=['box'], help='gru: subset of box cwh vel speed')
    ap.add_argument('--visual', nargs='*', default=[], help='gru: pooled visual streams, e.g. local_rgb global_rgb')
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=None, help='default: gru 0.0, acit 0.3')
    ap.add_argument('--l2_head', type=float, default=1e-3, help='acit: L2 on the MLP head')
    ap.add_argument('--optim', choices=['adam', 'adamw'], default=None, help='default: gru adamw, acit adam')
    ap.add_argument('--lr', type=float, default=None, help='default: gru 1e-3, acit 2e-5')
    ap.add_argument('--weight_decay', type=float, default=None, help='default: gru 1e-2 (adamw), acit 0')
    ap.add_argument('--batch_size', type=int, default=None, help='GLOBAL batch size; default gru 64, acit 16')
    ap.add_argument('--epochs', type=int, default=None, help='default: gru 100, acit 40')
    ap.add_argument('--patience', type=int, default=10, help='early stopping on val AUC')
    ap.add_argument('--class_weights', type=int, default=None,
                    help='1: benchmark weights (neg=pos/N, pos=neg/N). default: gru 0, acit 1 (paper)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--no_amp', action='store_true')
    args = ap.parse_args()
    defaults = {'gru': dict(dropout=0.0, optim='adamw', lr=1e-3, weight_decay=1e-2, batch_size=64, epochs=100, class_weights=0),
                'acit': dict(dropout=0.3, optim='adam', lr=2e-5, weight_decay=0.0, batch_size=16, epochs=40, class_weights=1)}
    for k, v in defaults[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    # ---- distributed setup (torchrun sets these; plain `python` = single process) ----
    ddp = 'LOCAL_RANK' in os.environ
    rank, world = (int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])) if ddp else (0, 1)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    if ddp:
        dist.init_process_group('nccl')
    is_main = rank == 0
    amp = not args.no_amp
    set_seed(args.seed + rank)

    # ---- data ----
    data = load_samples(args.samples)
    inputs = model_inputs(args)
    stats = kin_stats_from(data['train'], args.kin_feats) if 'kin' in inputs else None
    ds = {s: JAADDataset(data[s], inputs, args.kin_feats, stats, args.feats) for s in ['train', 'val', 'test']}
    sampler = DistributedSampler(ds['train'], world, rank, shuffle=True, seed=args.seed) if ddp else None
    per_gpu = args.batch_size // world
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(ds['train'], per_gpu, shuffle=sampler is None, sampler=sampler,
                              drop_last=len(ds['train']) >= args.batch_size, **loader_kw)
    eval_loaders = {s: DataLoader(ds[s], 2 * per_gpu, shuffle=False, **loader_kw) for s in ['val', 'test']}

    y_train = data['train']['label']
    n_pos, n_tot = int(y_train.sum()), len(y_train)
    w_neg, w_pos = (n_pos / n_tot, (n_tot - n_pos) / n_tot) if args.class_weights else (1.0, 1.0)

    # ---- model ----
    model = build_model(args, input_dims(inputs, args.kin_feats)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    raw = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    opt_cls = torch.optim.AdamW if args.optim == 'adamw' else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    if is_main:
        os.makedirs(args.out, exist_ok=True)
        cfg = dict(vars(args), inputs=inputs, n_params=n_params, world_size=world, protocol=data['meta'],
                   class_weights_used=[w_neg, w_pos], train_counts=dict(pos=n_pos, neg=n_tot - n_pos))
        with open(os.path.join(args.out, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=1)
        print(f'{args.model}: {n_params / 1e6:.3f}M params | inputs {inputs} | train {n_tot} (pos {n_pos}) '
              f'| val {len(ds["val"])} | test {len(ds["test"])} | world {world}', flush=True)
        hist = open(os.path.join(args.out, 'history.csv'), 'w', newline='')
        hist_w = csv.writer(hist)
        hist_w.writerow(['epoch', 'train_loss', 'val_auc', 'val_bench_acc', 'val_bench_f1', 'seconds'])

    # ---- train ----
    best_auc, bad_epochs = -1.0, 0
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        tot, cnt = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            y = batch['label']
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                logits = model(batch)
            weight = torch.where(y > 0.5, w_pos, w_neg)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y, weight=weight)
            if hasattr(raw, 'reg_loss'):
                loss = loss + raw.reg_loss()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            cnt += len(y)

        # validation on rank 0 only; the stop decision is broadcast so all ranks leave the loop together
        stop = torch.zeros(1, device=device)
        if is_main:
            yv, pv = predict(raw, eval_loaders['val'], device, amp)
            m = compute_metrics(yv, pv)
            score = m['auc'] if np.isfinite(m['auc']) else -np.inf  # nan if val has a single class
            improved = epoch == 0 or score > best_auc
            if improved:
                best_auc, bad_epochs = score, 0
                torch.save(raw.state_dict(), os.path.join(args.out, 'best.pt'))
            else:
                bad_epochs += 1
            hist_w.writerow([epoch, tot / cnt, m['auc'], m['bench_acc'], m['bench_f1'], round(time.time() - t0, 1)])
            hist.flush()
            print(f'ep {epoch:3d} loss {tot / cnt:.4f} | val auc {m["auc"]:.3f} acc {m["bench_acc"]:.3f} '
                  f'f1 {m["bench_f1"]:.3f} {"*" if improved else ""} | {time.time() - t0:.0f}s', flush=True)
            stop[0] = float(bad_epochs >= args.patience)
        if ddp:
            dist.broadcast(stop, 0)
        if stop.item():
            break

    # ---- final evaluation with the best checkpoint (rank 0) ----
    if is_main:
        hist.close()
        raw.load_state_dict(torch.load(os.path.join(args.out, 'best.pt'), map_location=device))
        yv, pv = predict(raw, eval_loaders['val'], device, amp)
        thr = select_threshold(yv, pv)
        yt, pt = predict(raw, eval_loaders['test'], device, amp)
        metrics = {'val': compute_metrics(yv, pv, thr), 'test': compute_metrics(yt, pt, thr), 'n_params': n_params,
                   'latency_ms_bs1': latency_ms(raw, next(iter(eval_loaders['test'])), device, amp),
                   'gpu': torch.cuda.get_device_name(device)}
        with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=1)
        save_predictions(os.path.join(args.out, 'val_predictions.csv'), data['val'], yv, pv)
        save_predictions(os.path.join(args.out, 'test_predictions.csv'), data['test'], yt, pt)
        t = metrics['test']
        print(f'TEST  acc {t["bench_acc"]:.3f} auc(bench) {t["bench_auc"]:.3f} f1 {t["bench_f1"]:.3f} '
              f'prec {t["bench_precision"]:.3f} rec {t["bench_recall"]:.3f} | true auc {t["auc"]:.3f} '
              f'| val-thr {thr:.3f}: bal_acc {t["thr_balanced_acc"]:.3f} f1 {t["thr_f1"]:.3f}', flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
