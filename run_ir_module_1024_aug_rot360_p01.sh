#!/bin/bash
# PatchCore 1024x1024 with denser rotation augmentation (full 360° coverage).
# Augmentations: rot90/180/270, 8x random rot ±180deg (uniform 0-360°), brightness/contrast jitter, gaussian noise.
# Training set: 13 originals + 169 augmented = 182 images.
# Coreset: approx_greedy_coreset -p 0.1 reduces ~2.98M patch features to ~298k.

DATA_PATH="/work/xxjustin77xx/Hengfeng_Patchcore/data/640C/ir_module_1024"
RESULTS_PATH="/work/xxjustin77xx/Hengfeng_Patchcore/results"

cd /work/xxjustin77xx/Hengfeng_Patchcore

python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group ir_module_WR50_L2-3_PS3_1024_aug_rot360_p0.1 \
  --log_project 640C \
  "$RESULTS_PATH" \
  patch_core \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1024 \
    --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 \
    --patchsize 3 \
  sampler \
    -p 0.1 \
    approx_greedy_coreset \
  dataset \
    --resize 1024 \
    --imagesize 1024 \
    --batch_size 1 \
    --num_workers 4 \
    -d ir_module \
    mvtec "$DATA_PATH"
