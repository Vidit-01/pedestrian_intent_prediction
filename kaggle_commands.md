# Kaggle notebook — cell by cell

Ready-made notebooks with these steps: `kaggle/baseline_a_gru/baseline_a_gru.ipynb`, `kaggle/baseline_b_acit/baseline_b_acit.ipynb`.

**Notebook settings:** Accelerator **GPU T4 x2** · Internet **On** · Add Input → dataset **`christopheronyiuke/jaad-video`**
(346 official JAAD mp4s, mounted at `/kaggle/input/jaad-video/JAAD_clips`).

Disk: `/kaggle/working` is capped at 20 GB (and is what "Save Version" keeps); big feature caches go to `/tmp`.
Feature cache sizes (8x8x1024 fp16 Swin maps, 5 modalities): **JAAD_beh ≈ 9.4 GB**, **JAAD_all ≈ 30 GB**.

---

### 1. Clone the code
```bash
!git clone https://github.com/<YOUR_USER>/<YOUR_REPO>.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -U timm      # swinv2 weights; transformers/torchvision/opencv are preinstalled
!nvidia-smi -L && df -h /kaggle/working /tmp
```

### 2. Dataset (model-agnostic, reuse for any future model)
```bash
# official annotations + split_ids + jaad_data.py from GitHub; clips symlinked from the Kaggle input (no download)
!python download_jaad.py --root /kaggle/working/JAAD --clips_from /kaggle/input/jaad-video/JAAD_clips
# no Kaggle dataset? fallback: --clips  (downloads the official 3.1 GB zip from York)
```

### 3. Samples + split (PCPA benchmark protocol: obs 16, TTE 30-60 frames, overlap 0.8, official video split)
```bash
!python build_jaad_samples.py --jaad_root /kaggle/working/JAAD --out /kaggle/working/samples --sample_type beh
!python build_jaad_samples.py --jaad_root /kaggle/working/JAAD --out /kaggle/working/samples --sample_type all
```
Expected: beh train/val/test = 2134/242/1881 samples, all = 8613/1265/6732. Manifests: `samples/jaad_*_manifest.csv`.

### 4. Controls: majority class + last-frame logistic regression (CPU)
```bash
!for S in beh all; do python simple_baselines.py --samples /kaggle/working/samples/jaad_$S.pkl --out runs/simple_$S; done
```

### 5. Baseline A — kinematic ablations 1-3, 3 seeds, both GPUs busy (tiny model: one process per GPU, no DDP)
```bash
%%bash
cd /kaggle/working/repo && mkdir -p logs
for SUB in beh all; do
  for FEATS in "box" "box speed" "box speed vel"; do
    TAG=${SUB}_$(echo $FEATS | tr ' ' '_')
    for SEED in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$((SEED % 2)) python train.py --model gru --samples /kaggle/working/samples/jaad_$SUB.pkl \
        --kin_feats $FEATS --seed $SEED --out runs/gru_${TAG}_s$SEED > logs/gru_${TAG}_s$SEED.log 2>&1 &
    done
    wait
    echo "== gru $TAG"; python evaluate.py runs/gru_${TAG}_s{0,1,2}
  done
done
```

### 6. Visual features (needed for ACIT and Baseline A ablations 4-5) — RAFT + SegFormer-B5 + Swin-V2-B, sharded over 2 GPUs
```bash
# JAAD_beh only (start here). Resumable: re-run the same command after a crash/timeout, finished videos are skipped.
!torchrun --nproc_per_node=2 extract_jaad_visual.py --jaad_root /kaggle/working/JAAD \
    --samples /kaggle/working/samples/jaad_beh.pkl --out /tmp/jaad_feats --vis 2
# JAAD_all (superset of beh; ~30 GB, check `df -h /tmp` first):
# !torchrun --nproc_per_node=2 extract_jaad_visual.py --jaad_root /kaggle/working/JAAD \
#     --samples /kaggle/working/samples/jaad_all.pkl --out /tmp/jaad_feats --vis 2
```
To extract once and reuse across sessions: write the beh cache to `--out /kaggle/working/jaad_feats` instead (fits under
20 GB), Save Version, then create a Kaggle dataset from the notebook output and pass `--feats /kaggle/input/<that-dataset>/jaad_feats` later.

### 7. Look at what Swin sees (local RGB | local flow | global RGB | global seg | global flow)
```python
from IPython.display import Image, display
import glob
for p in sorted(glob.glob('/tmp/jaad_feats/vis/*.jpg'))[:6]:
    print(p); display(Image(p))
```

### 8. Baseline B — ACIT, DDP on both T4s, 3 seeds
```bash
%%bash
cd /kaggle/working/repo && export NCCL_P2P_DISABLE=1   # avoids NCCL P2P hangs seen on some T4 x2 hosts
for SEED in 0 1 2; do
  torchrun --nproc_per_node=2 train.py --model acit --samples /kaggle/working/samples/jaad_beh.pkl \
    --feats /tmp/jaad_feats --seed $SEED --out runs/acit_beh_s$SEED 2>&1 | tee logs/acit_beh_s$SEED.log | grep -E "^(acit|ep|TEST)"
done
python evaluate.py runs/acit_beh_s{0,1,2}
```
(JAAD_all: same with `jaad_all.pkl` once the all-features are extracted.)

### 9. Baseline A — visual ablations 4-5 (pooled Swin features of the crop / full scene)
```bash
%%bash
cd /kaggle/working/repo
for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python train.py --model gru --samples /kaggle/working/samples/jaad_beh.pkl --feats /tmp/jaad_feats \
    --kin_feats box speed vel --visual local_rgb --seed $SEED --out runs/gru_beh_crop_s$SEED > logs/gru_beh_crop_s$SEED.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python train.py --model gru --samples /kaggle/working/samples/jaad_beh.pkl --feats /tmp/jaad_feats \
    --kin_feats box speed vel --visual local_rgb global_rgb --seed $SEED --out runs/gru_beh_crop_scene_s$SEED > logs/gru_beh_crop_scene_s$SEED.log 2>&1 &
  wait
done
python evaluate.py runs/gru_beh_crop_s{0,1,2}; python evaluate.py runs/gru_beh_crop_scene_s{0,1,2}
```

### 10. Keep the results
```bash
!cd /kaggle/working/repo && zip -qr /kaggle/working/runs.zip runs logs -x "*.pt" && ls -lh /kaggle/working/runs.zip
```

---
**Reading the numbers.** `bench_*` = PCPA-benchmark/ACIT-paper style at threshold 0.5; `bench_auc` is ROC-AUC of the
*rounded* predictions (that's what the published tables report). `auc` is the real ROC-AUC. `thr_*` use a threshold
picked on validation. Paper targets for ACIT: JAAD_beh acc .70 / AUC .62 / F1 .79 / P .69 / R .93;
JAAD_all .89 / .83 / .69 / .65 / .74.
