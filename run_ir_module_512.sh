#!/bin/bash
# Run PatchCore on IR module PCB images at 512x512 resolution, layer2+3.

DATA_PATH="/work/xxjustin77xx/patchcore-inspection/data/ir_module_512"
RESULTS_PATH="/work/xxjustin77xx/patchcore-inspection/results"

cd /work/xxjustin77xx/patchcore-inspection

python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group ir_module_WR50_L2-3_PS3_512 \
  --log_project IR_Module \
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
    -p 1.0 \
    identity \
  dataset \
    --resize 512 \
    --imagesize 512 \
    --batch_size 1 \
    --num_workers 4 \
    -d ir_module \
    mvtec "$DATA_PATH"
