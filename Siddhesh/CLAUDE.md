# CLAUDE.md

Pedestrian crossing-intention models on JAAD: dense vs routed PedViViT-style ViT (Kaggle 2x T4), Pedestrian Graph+ and GTransPDM (local GPU).
Overview, papers and results: `README.md`. Full record: `docs/RESULTS.md`. Routing spec: `docs/idea.md`. Kaggle run sheet: `kaggle_commands.md`.

## Layout
- `routed_vit/` ViT package (model, data, CV folds `cv.py`, training `train.py`, recipes `recipes.py`, routing analysis `analysis.py`, reports).
- `pedgraph/` Pedestrian Graph+ and GTransPDM (`fastdata.py` GPU data, `net.py`, `gtranspdm.py`, `cvfit.py` CV trainer, `hpo.py` Optuna, `final.py`, `run_gtrans.py`).
- `build_notebooks.py` generates `kaggle/A|B|C/*.ipynb` (embeds the package source). **Rerun after editing any package file.**
- `results/round1|round2/` outputs (CSV/JSON/npz; checkpoints are git-ignored). `data/` and `logs/` are git-ignored.

## Pipeline (each step is a CLI; the sample builder lives in the base repo, one level above this folder)
- Samples: `routed_vit.data.build_samples(repo_root, JAAD, out.pkl, 'beh'|'all')` -> `data/samples_ctx/jaad_{beh,all}.pkl` (PCPA windows + context labels).
- Poses: `python -m pedgraph.extract --samples <pkl> --clips data/JAAD/JAAD_clips --out data/pose_{beh,all}` (ViTPose-B).
- ViT: notebooks A then B (`routed_vit.launcher.run_jobs` runs one process per GPU).
- Pedestrian Graph+: `python -m pedgraph.hpo ...` then `python -m pedgraph.final ...`; GTransPDM: `python -m pedgraph.run_gtrans ...` (commands in `pedgraph/README.md`).

## Protocol and gotchas
- Round 2 protocol: train+val pooled, 5 video-grouped folds (`routed_vit/cv.py`, same for every model), fixed epoch budget, EMA, OOF AUC for selection, test split only predicted. Do not select anything on test.
- `bench_auc` = AUC of rounded predictions (benchmark convention); `auc` is the real one. JAAD_all test is 17.5% positive (majority accuracy 0.825), JAAD_beh test 62.6% positive: read AUC / balanced accuracy, not accuracy.
- Dense and B-soft must give identical outputs at init (max difference 0.0, checked in notebook B). Ego speed is deliberately not a ViT input.
- Pedestrian Graph+ is built from the authors' code (paper not read); GTransPDM from the arXiv paper (no code released); assumptions are in `pedgraph/gtranspdm.py`.
- Optuna journal storage uses `JournalFileOpenLock` (symlink locks need admin rights on Windows). Searches are resumable: rerun the same command (`scripts/resume_pg_all.sh`).
- Windows: set `PYTHONUTF8=1` when reading Kaggle logs with the CLI.
