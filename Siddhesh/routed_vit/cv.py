"""Cross-validation protocol shared by the ViT study and Pedestrian Graph+.

Why: the official JAAD_beh validation split has 22 pedestrians. Selecting epochs / hyper-parameters on it is close to random
(val AUC 0.8 vs test AUC 0.6 in the first round of experiments). Instead, train+val are pooled (216 pedestrians) and split into
K folds grouped by *video* (pedestrians of one video share the scene, so they must not straddle train and validation).
The official test split is never used for any selection; it is only predicted by the fold models and averaged.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def pool_trainval(data):
    """Concatenate the 'train' and 'val' records of a samples pickle into one record (rows: train first, then val)."""
    a, b = data['train'], data['val']
    out = {}
    for k in a:
        out[k] = a[k] + b[k] if isinstance(a[k], list) else np.concatenate([a[k], b[k]])
    return out


def ped_groups(rec):
    return np.array([f'{v}/{p}' for v, p in zip(rec['vid'], rec['pid'])])


def make_folds(rec, n_folds=5, seed=0):
    """List of (train_rows, val_rows) over the pooled record. Grouped by video, stratified by label, deterministic in `seed`.
    The fold assignment is independent of the training seed and of the model variant, so every model sees identical folds."""
    vid = np.array(rec['vid'])
    y = np.asarray(rec['label'])
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(np.sort(tr), np.sort(va)) for tr, va in skf.split(np.zeros(len(y)), y, vid)]


def describe_folds(rec, folds):
    g, y = ped_groups(rec), np.asarray(rec['label'])
    rows = []
    for i, (tr, va) in enumerate(folds):
        rows.append({'fold': i, 'train_windows': len(tr), 'val_windows': len(va), 'train_peds': len(set(g[tr])),
                     'val_peds': len(set(g[va])), 'val_pos_rate': round(float(y[va].mean()), 3)})
    return rows
