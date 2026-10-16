#!/bin/bash
# Resume Pedestrian Graph+ on JAAD_all (full settings: 120-trial search, re-score, image stage, final evaluation).
# Finished trials and stages are kept in results/round2/pedgraph_all/ (Optuna journal), so this continues where it stopped.
# Run from anywhere:  bash scripts/resume_pg_all.sh     (needs data/samples_ctx/jaad_all.pkl and data/pose_all/, see README)
cd "$(dirname "$0")/.."
mkdir -p logs
A=data/samples_ctx/jaad_all.pkl; AC=data/pose_all; OUT=results/round2/pedgraph_all
python -m pedgraph.hpo --samples $A --cache $AC --out $OUT --trials 120 --parallel 3 >> logs/hpo_all.log 2>&1
python -m pedgraph.final --samples $A --cache $AC --hpo $OUT --out $OUT/final --seeds 5 --baseline_seeds 3 \
       --ablate --ablation_seeds 3 --ensemble_topk 3 > logs/final_all.log 2>&1
echo finished > logs/pg_all.done
