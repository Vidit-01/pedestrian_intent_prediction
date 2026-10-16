"""Notebook-level helpers: load trained runs, result tables with CIs, and the per-run analysis drivers + plots."""
import json
import os

import numpy as np
import pandas as pd
import torch

from . import analysis as an
from .data import CTX_LEVELS, CTX_NAMES, CropDataset, groups_of
from .metrics import bootstrap_metrics
from .model import RoutedViT


def load_run(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    m = RoutedViT(cfg['variant'], L0=cfg['L0'], k=cfg['k'], use_box=not cfg.get('no_box', False),
                  box_only=cfg.get('box_only', False))
    sd = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu')
    m.load_state_dict({k: v.float() if v.dtype == torch.float16 else v for k, v in sd.items()})
    return m.to(device).eval(), cfg


def load_outputs(run_dir, split='test'):
    return dict(np.load(os.path.join(run_dir, f'{split}_outputs.npz')))


def rdir(root, variant, seed, fold):
    return os.path.join(root, f'{variant}_s{seed}_f{fold}')


def is_done(root, variant, seed, fold):
    return os.path.exists(os.path.join(rdir(root, variant, seed, fold), 'metrics.json'))


def collect(root, variant, seeds, folds, n_pool, n_test):
    """Per training seed: out-of-fold probabilities over the pooled train+val record (NaN where a fold is missing) and the
    fold-ensembled (mean) test probabilities. -> {seed: {'oof': [n_pool], 'test': [n_test], 'folds_done': int, 'fold_auc': [..]}}"""
    from sklearn.metrics import roc_auc_score
    res = {}
    for s in seeds:
        oof, tst, k, fold_auc = np.full(n_pool, np.nan), np.zeros(n_test), 0, []
        for f in folds:
            if not is_done(root, variant, s, f):
                continue
            d = rdir(root, variant, s, f)
            vo, to = load_outputs(d, 'val'), load_outputs(d, 'test')
            oof[vo['idx']] = vo['p']
            fold_auc.append(roc_auc_score(vo['y'], vo['p']))
            tst[to['idx']] += to['p']
            k += 1
        if k:
            res[s] = {'oof': oof, 'test': tst / k, 'folds_done': k, 'fold_auc': fold_auc}
    return res


def results_table(root, variants, seeds, folds, data, n_boot=500):
    """Cross-validated results. Per (variant, seed): out-of-fold AUC on the pooled 216 train+val pedestrians (selection-free:
    every window is predicted by a model that never saw its video), per-fold AUC mean +- std, and the test AUC of the fold
    ensemble. The decision threshold for acc/F1 is picked on the OUT-OF-FOLD predictions (balanced accuracy), never on test.
    A final row per variant ('ens') averages all seeds and folds, with pedestrian-cluster bootstrap CIs on test."""
    from sklearn.metrics import roc_auc_score
    from .cv import pool_trainval
    from .metrics import compute_metrics, select_threshold
    pool = pool_trainval(data)
    y_pool, y_te = np.asarray(pool['label']), np.asarray(data['test']['label'])
    g_te = groups_of(data['test'])
    maj = max(y_te.mean(), 1 - y_te.mean())
    rows = []
    for v in variants:
        col = collect(root, v, seeds, folds, len(y_pool), len(y_te))
        for s, c in col.items():
            full = c['folds_done'] == len(folds)
            m_oof = ~np.isnan(c['oof'])
            thr = select_threshold(y_pool[m_oof], c['oof'][m_oof]) if full else 0.5
            mt = compute_metrics(y_te, c['test'], thr)
            rows.append({'variant': v, 'seed': s, 'folds': c['folds_done'],
                         'oof_auc': roc_auc_score(y_pool[m_oof], c['oof'][m_oof]),
                         'fold_auc': f"{np.mean(c['fold_auc']):.3f}±{np.std(c['fold_auc']):.3f}",
                         'test_auc': mt['auc'], 'bench_acc': mt['bench_acc'], 'bench_f1': mt['bench_f1'],
                         'thr': thr, 'thr_bacc': mt['thr_balanced_acc'], 'thr_f1': mt['thr_f1'], 'bench_auc': mt['bench_auc'],
                         'brier': mt['brier']})
        if col:
            p_ens = np.mean([c['test'] for c in col.values()], 0)
            oof_ens = np.nanmean([c['oof'] for c in col.values()], 0)
            m_oof = ~np.isnan(oof_ens)
            thr = select_threshold(y_pool[m_oof], oof_ens[m_oof])
            ci = bootstrap_metrics(y_te, p_ens, g_te, thr, n_boot=n_boot, keys=('auc', 'bench_acc', 'bench_f1', 'thr_f1'))
            mt = compute_metrics(y_te, p_ens, thr)
            rows.append({'variant': v, 'seed': 'ens', 'folds': sum(c['folds_done'] for c in col.values()),
                         'oof_auc': roc_auc_score(y_pool[m_oof], oof_ens[m_oof]), 'fold_auc': '',
                         'test_auc': mt['auc'], 'bench_acc': mt['bench_acc'], 'bench_f1': mt['bench_f1'], 'thr': thr,
                         'thr_bacc': mt['thr_balanced_acc'], 'thr_f1': mt['thr_f1'], 'bench_auc': mt['bench_auc'],
                         'brier': mt['brier'], 'test_auc_ci': '[%.3f, %.3f]' % ci['auc'][1:],
                         'bench_acc_ci': '[%.3f, %.3f]' % ci['bench_acc'][1:], 'thr_f1_ci': '[%.3f, %.3f]' % ci['thr_f1'][1:]})
    df = pd.DataFrame(rows)
    print(f'test majority-class accuracy {maj:.3f} (positive rate {y_te.mean():.3f}): acc/F1@0.5 below that carry no signal; read AUC.')
    if not df.empty:
        per = df[df.seed != 'ens']
        print('mean ± std over training seeds:')
        print(per.groupby('variant')[['oof_auc', 'test_auc', 'thr_bacc', 'thr_f1']].agg(['mean', 'std']).round(3).to_string())
    return df


def ensemble_test_probs(root, variant, seeds, folds, n_pool, n_test):
    """One fold-ensembled test-probability vector per finished seed (input of the paired H1 bootstrap)."""
    col = collect(root, variant, seeds, folds, n_pool, n_test)
    return [c['test'] for c in col.values() if c['folds_done'] == len(folds)]


def plot_history(root, variants, seeds, folds):
    """Mean +- min/max over folds and seeds of the per-epoch validation AUC (EMA model) and training loss."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for v in variants:
        hs = [pd.read_csv(os.path.join(rdir(root, v, s, f), 'history.csv')) for s in seeds for f in folds if is_done(root, v, s, f)]
        if not hs:
            continue
        for a, col in zip(ax, ['val_auc', 'train_task']):
            M = np.stack([h[col].values for h in hs])
            x = hs[0].epoch.values
            a.plot(x, M.mean(0), label=f'{v} (n={len(hs)})')
            a.fill_between(x, M.min(0), M.max(0), alpha=.15)
    ax[0].set_title('held-out-fold AUC (mean, min-max band)')
    ax[1].set_title('train loss (task)')
    ax[0].legend(fontsize=7)
    for a in ax:
        a.set_xlabel('epoch')
    plt.tight_layout()
    plt.show()


def heatmap(mat, title, xlabel='head', ylabel='layer', ax=None, ylabels=None, vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(mat.shape[1]))
    if ylabels is not None:
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax)


# --------------------------------------------------------------------------- Phase A / H5 probes (any variant)
def head_probes(run_dir, data, crops, device, n_samples=300, seed=0, importance=True):
    """Head importance (mean ablation) and gradient norms on a fixed random test subset; saved to <run_dir>/probes.npz."""
    out = os.path.join(run_dir, 'probes.npz')
    if os.path.exists(out):
        return dict(np.load(out))
    model, cfg = load_run(run_dir, device)
    ds = CropDataset(data['test'], crops)
    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(len(ds), min(n_samples, len(ds)), replace=False))
    routed_layers = list(range(model.L0, len(model.blocks)))
    res = {'idxs': idxs, 'grad_norms': an.gradient_norms(model, ds, idxs, device, routed_layers)}
    if importance:
        imp, base = an.head_importance(model, ds, idxs, device)
        res.update(importance=imp, base_logit=base)
    np.savez_compressed(out, **res)
    del model
    torch.cuda.empty_cache()
    return res


def grad_entropy_table(probe, ctx_test, names=CTX_NAMES):
    """Mean (over routed layers x heads) normalised entropy of the gradient share per context variable (H5)."""
    ctx = ctx_test[probe['idxs']]
    rows = {}
    for k, n in enumerate(names):
        if len(np.unique(ctx[:, k])) < 2:
            continue
        rows[n] = float(an.gradient_share_entropy(probe['grad_norms'], ctx[:, k]).mean())
    return rows


def importance_by_context(probe, ctx_test, name):
    """Mean head importance per context level: dict level -> [L,H]."""
    k = CTX_NAMES.index(name)
    ctx = ctx_test[probe['idxs']][:, k]
    return {CTX_LEVELS[name][b]: probe['importance'][ctx == b].mean(0) for b in np.unique(ctx)}
