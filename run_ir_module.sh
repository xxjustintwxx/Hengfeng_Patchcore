#!/bin/bash
# Run PatchCore on IR module PCB images.
# Data lives locally at data/ir_module/ir_module/{train/good, test/good, test/defect, ground_truth/defect}

DATA_PATH="/work/xxjustin77xx/Hengfeng_Patchcore/data/ir_module_256"
RESULTS_PATH="/work/xxjustin77xx/Hengfeng_Patchcore/results"

cd /work/xxjustin77xx/Hengfeng_Patchcore

python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group ir_module_WR50_L2_PS3 \
  --log_project IR_Module \
  "$RESULTS_PATH" \
  patch_core \
    -b wideresnet50 \
    -le layer2 \
    --pretrain_embed_dimension 512 \
    --target_embed_dimension 512 \
    --anomaly_scorer_num_nn 1 \
    --patchsize 3 \
  sampler \
    -p 1.0 \
    identity \
  dataset \
    --resize 256 \
    --imagesize 224 \
    --batch_size 1 \
    --num_workers 4 \
    -d ir_module \
    mvtec "$DATA_PATH"
