# CLAUDE.md

Pedestrian crossing-intention baselines on JAAD (Baseline A: GRU, Baseline B: ACIT), run on Kaggle 2x T4.
Spec: `Vidit.md` (a Python wrapper around the markdown spec). Kaggle run sheet: `kaggle_commands.md`.
Kaggle notebooks: `kaggle/baseline_a_gru/`, `kaggle/baseline_b_acit/` (each has kernel-metadata.json; push with `kaggle kernels push -p <dir>`).

## Pipeline (each step is a standalone CLI)
- `python download_jaad.py --root JAAD [--clips_from DIR | --clips]` — clones official JAAD repo (annotations, split_ids, jaad_data.py)
- `python build_jaad_samples.py --jaad_root JAAD --out samples --sample_type beh|all` — PCPA benchmark protocol
- `torchrun --nproc_per_node=2 extract_jaad_visual.py --jaad_root JAAD --samples samples/jaad_beh.pkl --out FEATS` — frozen Swin features
- `python train.py --model gru|acit --samples ... [--feats FEATS] --out runs/x` (single GPU) or via `torchrun --nproc_per_node=2` (DDP)
- `python evaluate.py runs/a runs/b ...` — mean±std over seeds; `python simple_baselines.py` — majority / logreg controls

## Quick local check (no visual features needed)
`python train.py --model gru --samples <samples>/jaad_all.pkl --kin_feats box speed vel --out /tmp/r --epochs 5 --num_workers 0`

## Gotchas
- Protocol must match PedestrianActionBenchmark: overlap windows are applied to val/test too (stride 3), min track 76.
- `bench_auc` = roc_auc on rounded probs (benchmark/ACIT convention); `auc` is the real one. Report both.
- JAAD has no numeric speed: `speed` = vehicle action category 0..4 (benchmark does the same).
- T4: no bf16 -> fp16 autocast + GradScaler. Val/test eval runs on rank 0 only; stop flag is broadcast.
- ACIT unspecified sizes (AMMI heads/FFN, TFA head_dim, MLP hidden) are chosen to give 5.20M params (paper 5.15M); see `models/acit.py` docstring.
- Feature caches: `<FEATS>/<modality>/<vid>.npy` [n,64,1024] fp16 + `index/<vid>.json` (row keys). JAAD_beh ~9.4 GB, JAAD_all ~30 GB.
- New models: add a branch in `train.py::build_model` / `model_inputs`; datasets yield any of `kin`, `speed`, `box`, `<modality>`, `<modality>:pool`.
