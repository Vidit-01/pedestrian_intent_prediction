"""Fine-tuning recipes for the dense-ViT recipe search (notebook A). Round 1 (lr 5e-5, no layer decay, early stopping on 22
validation pedestrians) peaked at epoch 1 and then memorised the training set. These four trade off how far the pretrained
ViT may move (LR, layer-wise decay, frozen prefix) against regularisation strength. Notebook A runs all of them with 5-fold
video-grouped CV (seed 42), picks the one with the best pooled out-of-fold AUC, and notebook B re-uses that exact recipe.
Every other training argument is the default of `routed_vit.train`.
"""
RECIPES = {
    'r1_lr2e-5': ['--epochs', 10, '--lr', 2e-5, '--llrd', 0.8],
    'r2_lr1e-5_long': ['--epochs', 14, '--lr', 1e-5, '--llrd', 0.85],
    'r3_frozen6': ['--epochs', 10, '--lr', 5e-5, '--llrd', 0.9, '--freeze_blocks', 6],
    'r4_strong_reg': ['--epochs', 8, '--lr', 2e-5, '--llrd', 0.75, '--drop_path', 0.2, '--label_smooth', 0.1,
                      '--box_drop', 0.3, '--box_noise', 0.04],
}
DEFAULT_RECIPE = 'r1_lr2e-5'

# B-sparse collapsed in round 1 (every clip picked the same 3 heads). Untested guesses to break that: longer soft phase,
# lower router LR, stronger balancing loss. Notebook B prints the routing diagnostics so they can be judged.
SPARSE_EXTRA = ['--sparse_start', 0.4, '--router_lr_mult', 5, '--lam_bal', 0.05]
