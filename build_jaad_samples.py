"""Build JAAD crossing-prediction samples with the PCPA benchmark protocol (Kotseruba et al., WACV 2021),
which is also the protocol ACIT follows.

Protocol (matches PedestrianActionBenchmark/action_predict.py::get_data_sequence + train_test.py):
  - tracks: jaad_data.JAAD.generate_data_trajectory_sequence(seq_type='crossing', subset='default',
    data_split_type='default') -> official video split 177 train / 29 val / 117 test.
    A track ends at the crossing point (behavioural peds) or 3 frames before the end of the track (others).
  - sample_type: 'beh' (JAAD_beh, behaviour-annotated peds only) or 'all' (JAAD_all, + non-behavioural peds, label 0).
  - label: 1 if the pedestrian crosses (attributes['crossing'] > 0) else 0. Same label for every window of a track.
  - windows: obs_length=16 frames, time-to-event in [30, 60] frames (1-2 s @ 30 fps), stride int((1-0.8)*16)=3,
    applied to train, val AND test (the benchmark does this). min_track_size = 16 + 60 = 76.

Output: <out>/jaad_<sample_type>.pkl  and  <out>/jaad_<sample_type>_manifest.csv

    python build_jaad_samples.py --jaad_root /kaggle/working/JAAD --out /kaggle/working/samples --sample_type beh
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np


def make_windows(n, obs_length, tte, stride):
    """Start indices of the observation windows in a track of length n (benchmark rule)."""
    start = n - obs_length - tte[1]
    end = n - obs_length - tte[0]
    return list(range(start, end + 1, stride))


def build(jaad_root, sample_type, obs_length=16, tte=(30, 60), overlap=0.8):
    sys.path.insert(0, jaad_root)
    from jaad_data import JAAD  # official parser, lives in the cloned repo

    imdb = JAAD(data_path=jaad_root)
    db = imdb.generate_database()
    opts = {'fstride': 1, 'sample_type': sample_type, 'subset': 'default', 'data_split_type': 'default',
            'seq_type': 'crossing', 'min_track_size': obs_length + tte[1],
            'height_rng': [0, float('inf')], 'squarify_ratio': 0}
    stride = max(1, int((1 - overlap) * obs_length)) if overlap > 0 else obs_length

    out = {'meta': {'dataset': 'jaad', 'sample_type': sample_type, 'obs_length': obs_length, 'tte': list(tte),
                    'overlap': overlap, 'stride': stride, 'fps': 30, 'subset': 'default', 'split': 'default'}}
    for split in ['train', 'val', 'test']:
        seqs = imdb.generate_data_trajectory_sequence(split, **opts)
        rec = {k: [] for k in ['vid', 'pid', 'frames', 'bbox', 'vehicle_act', 'occlusion', 'img_wh', 'label', 'tte']}
        for t in range(len(seqs['image'])):
            n = len(seqs['image'][t])
            vid = os.path.basename(os.path.dirname(seqs['image'][t][0]))
            pid = seqs['pid'][t][0][0]
            frames = np.array([int(os.path.splitext(os.path.basename(p))[0]) for p in seqs['image'][t]])
            bbox = np.array(seqs['bbox'][t], dtype=np.float32)
            veh = np.array([v[0] for v in seqs['vehicle_act'][t]], dtype=np.int64)
            occ = np.array(seqs['occlusion'][t], dtype=np.int64)
            label = int(seqs['activities'][t][0][0])
            wh = (db[vid]['width'], db[vid]['height'])
            for s in make_windows(n, obs_length, tte, stride):
                sl = slice(s, s + obs_length)
                rec['vid'].append(vid)
                rec['pid'].append(pid)
                rec['frames'].append(frames[sl])
                rec['bbox'].append(bbox[sl])
                rec['vehicle_act'].append(veh[sl])
                rec['occlusion'].append(occ[sl])
                rec['img_wh'].append(wh)
                rec['label'].append(label)
                rec['tte'].append(n - (s + obs_length))
        for k in ['frames', 'bbox', 'vehicle_act', 'occlusion', 'img_wh', 'label', 'tte']:
            rec[k] = np.stack(rec[k]) if k in ('frames', 'bbox', 'vehicle_act', 'occlusion') else np.array(rec[k])
        out[split] = rec
        pos = int(rec['label'].sum())
        print(f'[{sample_type}/{split}] videos={len(set(rec["vid"]))} peds={len(set(rec["pid"]))} '
              f'samples={len(rec["label"])} pos={pos} neg={len(rec["label"]) - pos}')

    # leakage check: a video must belong to exactly one split
    vids = {s: set(out[s]['vid']) for s in ['train', 'val', 'test']}
    assert not (vids['train'] & vids['val'] or vids['train'] & vids['test'] or vids['val'] & vids['test'])
    return out


def write_manifest(data, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'vid', 'pid', 'first_frame', 'last_frame', 'tte_frames', 'label'])
        for s in ['train', 'val', 'test']:
            r = data[s]
            for i in range(len(r['label'])):
                w.writerow([s, r['vid'][i], r['pid'][i], r['frames'][i][0], r['frames'][i][-1], r['tte'][i], r['label'][i]])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--jaad_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample_type', choices=['beh', 'all'], default='beh')
    ap.add_argument('--obs_length', type=int, default=16)
    ap.add_argument('--tte', type=int, nargs=2, default=[30, 60])
    ap.add_argument('--overlap', type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = build(args.jaad_root, args.sample_type, args.obs_length, tuple(args.tte), args.overlap)
    name = f'jaad_{args.sample_type}'
    with open(os.path.join(args.out, name + '.pkl'), 'wb') as f:
        pickle.dump(data, f)
    write_manifest(data, os.path.join(args.out, name + '_manifest.csv'))
    print('wrote', os.path.join(args.out, name + '.pkl'))
