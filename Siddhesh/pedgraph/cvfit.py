"""Cross-validated training / prediction for Pedestrian Graph+ (round 2).

Protocol (same as the ViT notebooks, see routed_vit/cv.py): train+val pooled (216 pedestrians), 5 video-grouped folds, fixed
epoch budget (no checkpoint picking), EMA weights, out-of-fold (OOF) AUC as the selection signal, the official test split only
predicted by the fold models. Round 1 trained once per seed on the official split and picked the best epoch on 22 validation
pedestrians (val AUC 0.47-0.88 from one epoch to the next), so it selected noise.

    cfg  = dict of hyper-parameters (DEFAULT_CFG / RELEASED_CFG / hpo.suggest)
    res  = run_cv(cfg, D, folds, seeds, device)  ->  {'oof':[S,n_pool], 'test':[S,n_test], 'fold_auc':[S][K], ...}
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .fastdata import build_batch, n_pose_channels, n_side_channels
from .net import build_model

# a strong, regularised starting point (the search explores around it)
DEFAULT_CFG = dict(
    arch='net', pose_repr='abs+rel', kp_vel=True, side='box+ego', use_image=False, fusion='gate',
    widths=[32, 64], kernel=5, dropout=0.25, adj_init='skeleton', head_hidden=0,
    epochs=40, bs=64, lr=3e-3, wd=1e-2, sched='onecycle', label_smooth=0.05, ema=0.99, clip=5.0, amp=False, class_weight=True,
    jitter=0.02, gscale=0.05, hflip=0.5, joint_drop=0.1, time_crop=0.25, img_aug=1.0, img_down=2, tta=True)

# the authors' released recipe (AdamW wd 1e-3, OneCycle 5e-3, 30 epochs, time crop, image + vehicle branches) inside the same
# CV pipeline. Augmentations are our GPU versions (the released PIL stack is replaced), poses are the 2D ViTPose ones.
RELEASED_CFG = dict(
    DEFAULT_CFG, arch='released', pose_repr='abs', kp_vel=False, side='ego_scalar', use_image=True, epochs=30, bs=32, lr=5e-3,
    wd=1e-3, label_smooth=0.0, ema=0.0, jitter=0.01, gscale=0.0, hflip=0.0, joint_drop=0.0, time_crop=0.5, tta=False, amp=True)


MAX_EPOCHS = None   # debug hook (hpo --max_epochs): caps the epoch budget of every fit


# GTransPDM (Xie et al., arXiv 2409.20223) trained with the paper's recipe: AdamW, constant lr 5e-5, batch 64, 32 epochs, wd 1e-4,
# unweighted cross-entropy (one logit here), no augmentation, no EMA, no TTA, no clipping. See gtranspdm.py for assumptions.
GTRANS_CFG = dict(
    DEFAULT_CFG, arch='gtranspdm', side='none', use_image=False, epochs=32, bs=64, lr=5e-5, wd=1e-4, sched='const',
    label_smooth=0.0, ema=0.0, clip=0.0, class_weight=False, jitter=0.0, gscale=0.0, hflip=0.0, joint_drop=0.0, time_crop=0.0,
    tta=False, img_aug=0.0)


def make_cfg(**kw):
    bad = set(kw) - set(DEFAULT_CFG)
    assert not bad, f'unknown config keys {bad}'
    return dict(DEFAULT_CFG, **kw)


class EMA:
    """EMA of parameters and floating-point buffers (BatchNorm statistics)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.tensors = [t for t in list(model.parameters()) + list(model.buffers()) if t.dtype.is_floating_point]
        self.shadow = [t.detach().clone() for t in self.tensors]

    @torch.no_grad()
    def update(self, step):
        d = min(self.decay, (1 + step) / (10 + step))
        torch._foreach_mul_(self.shadow, d)
        torch._foreach_add_(self.shadow, [t.detach() for t in self.tensors], alpha=1 - d)

    @torch.no_grad()
    def apply(self):
        for t, s in zip(self.tensors, self.shadow):
            t.copy_(s)


@torch.no_grad()
def predict_logits(model, D, rows, cfg, tta=None, chunk=512):
    model.eval()
    tta = cfg['tta'] if tta is None else tta
    out = []
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        views = [None, torch.ones(len(r), dtype=torch.bool, device=D.device)] if tta else [None]
        lo = 0
        for fl in views:
            pose, side, img = build_batch(D, r, cfg, train=False, flip=fl)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp'] and D.device.type == 'cuda'):
                lo = lo + model(pose, side, img).float().squeeze(1)
        out.append((lo / len(views)).cpu())
    return torch.cat(out).numpy()


def fit_predict(cfg, D, tr_rows, pred_rows, seed, track_rows=None):
    """Train on `tr_rows` for a fixed budget; return {name: probabilities} (EMA model at the end) and a per-epoch history.
    track_rows: optional rows scored after every epoch (diagnostic curves only; never used to stop or select)."""
    dev = D.device
    if MAX_EPOCHS:
        cfg = dict(cfg, epochs=min(cfg['epochs'], MAX_EPOCHS))
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    model = build_model(cfg, n_pose_channels(cfg), n_side_channels(cfg)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    tr_rows = np.asarray(tr_rows)
    bs = min(cfg['bs'], len(tr_rows))
    spe = len(tr_rows) // bs
    total = cfg['epochs'] * spe
    if cfg['sched'] == 'const':
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    elif cfg['sched'] == 'onecycle':
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], div_factor=10.0, final_div_factor=1e4, total_steps=total)
    else:   # linear warm-up (10%) then cosine to ~0
        w = max(1, int(0.1 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, total - w))) * 0.999 + 1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'] and dev.type == 'cuda')
    ema = EMA(model, cfg['ema']) if cfg['ema'] > 0 else None
    y_all = D.y[torch.as_tensor(tr_rows, device=dev)]
    pos = float(y_all.mean())
    w1, w0 = (0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6)) if cfg['class_weight'] else (1.0, 1.0)   # inverse class frequency
    hist, step = [], 0
    for ep in range(cfg['epochs']):
        model.train()
        perm = torch.randperm(len(tr_rows), device=dev, generator=gen).cpu().numpy()
        tot = 0.0
        for b in range(spe):
            rows = tr_rows[perm[b * bs:(b + 1) * bs]]
            pose, side, img = build_batch(D, rows, cfg, train=True, gen=gen)
            if cfg['time_crop'] > 0 and torch.rand(1, device=dev, generator=gen).item() < cfg['time_crop']:
                k = int(torch.randint(2, pose.shape[2], (1,), device=dev, generator=gen).item())
                pose = pose[:, :, -k:]                                 # released recipe: only the pose stream is shortened
            y = D.y[torch.as_tensor(rows, device=dev)]
            ys = y * (1 - cfg['label_smooth']) + 0.5 * cfg['label_smooth']
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp'] and dev.type == 'cuda'):
                lo = model(pose, side, img)
            wt = torch.where(y > 0.5, w1, w0)
            loss = (F.binary_cross_entropy_with_logits(lo.float().squeeze(1), ys, reduction='none') * wt).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg['clip'] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if ema:
                ema.update(step)
            tot += float(loss.detach())
        row = {'epoch': ep + 1, 'train_loss': tot / max(spe, 1)}
        if track_rows is not None:
            if ema:
                live = copy.deepcopy(model.state_dict())
                ema.apply()
            lo_t = predict_logits(model, D, track_rows, cfg)
            row['val_auc'] = roc_auc_score(D.y_np[track_rows], lo_t)
            if ema:
                model.load_state_dict(live)
        hist.append(row)
    if ema:
        ema.apply()
    preds = {k: 1 / (1 + np.exp(-predict_logits(model, D, np.asarray(r), cfg))) for k, r in pred_rows.items()}
    return preds, hist


def run_cv(cfg, D, folds, seeds, trial=None, track=False, verbose=False):
    """Fixed-budget K-fold CV. Returns dict(oof [S,n_pool] (each window predicted by a model that never saw its video),
    test [S,n_test] (fold-ensemble per seed), fold_auc [S][K], hist [S][K] (epoch curves), minutes).
    trial: optional optuna trial; reports the running mean fold AUC of the first seed and prunes."""
    t0 = time.time()
    S, n_pool, n_test = len(seeds), D.n_pool, D.n_test
    oof, test = np.full((S, n_pool), np.nan), np.zeros((S, n_test))
    fold_auc, hists = [[] for _ in seeds], [[] for _ in seeds]
    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            preds, h = fit_predict(cfg, D, tr, {'val': va, 'test': D.test_rows}, seed * 100 + fi, va if track else None)
            oof[si, va] = preds['val']
            test[si] += preds['test'] / len(folds)
            fold_auc[si].append(roc_auc_score(D.y_np[va], preds['val']))
            hists[si].append(h)
            if verbose:
                print(f'  seed {seed} fold {fi}: AUC {fold_auc[si][-1]:.3f}', flush=True)
            if trial is not None and si == 0:
                trial.report(float(np.mean(fold_auc[0])), fi)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
    test = np.clip(test, 0.0, 1.0)   # fold-sum rounding can leave 1.0000000x
    return {'oof': oof, 'test': test, 'fold_auc': fold_auc, 'hist': hists, 'minutes': (time.time() - t0) / 60}


def score(res):
    """Selection score: mean over seeds of the mean fold AUC (each fold AUC is on ~40 unseen pedestrians)."""
    return float(np.mean([np.mean(f) for f in res['fold_auc']]))
