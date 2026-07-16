#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --job-name=pc_colorjitter
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-04:00:00
#SBATCH --output=/work/xxjustin77xx/results/job_log/job-%j.out
#SBATCH --error=/work/xxjustin77xx/results/job_log/job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=justin135246j@gmail.com

module purge
module load cuda/12.4
module load gcc/11.5.0

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
mkdir -p /work/xxjustin77xx/tmp
export TMPDIR=/work/xxjustin77xx/tmp
export TEMP=/work/xxjustin77xx/tmp
export TMP=/work/xxjustin77xx/tmp
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

__conda_setup="$('/home/xxjustin77xx/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"
else
    if [ -f "/home/xxjustin77xx/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/xxjustin77xx/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/xxjustin77xx/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate aoi

cd /work/xxjustin77xx/Hengfeng_Patchcore

TRAIN_DIR="data/ir_module_1024/ir_module/train/good"

# Step 1: Remove existing aug images and generate color-jitter-only augmentations
# (no rotation: screw suppression now handled by YOLO)
echo "=== Step 1: Regenerate augmented training images (no rotation, strong color jitter) ==="
python -c "
import glob, os
aug_files = glob.glob('${TRAIN_DIR}/*_aug_*.jpg')
for f in aug_files:
    os.remove(f)
print(f'Removed {len(aug_files)} old aug files')
"

python augment_train.py \
  --src "$TRAIN_DIR" \
  --no_fixed_rot \
  --n_random_rot 0 \
  --n_jitter 6 \
  --strong_jitter \
  --n_noise 1 \
  --seed 42

echo ""
echo "=== Step 2: Train PatchCore (colorjitter aug, p=0.1) ==="
python bin/run_patchcore.py \
  --gpu 0 \
  --seed 0 \
  --save_patchcore_model \
  --log_group ir_module_WR50_L2-3_PS3_1024_colorjitter_p0.1 \
  --log_project IR_Module \
  results \
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
    mvtec "data/ir_module_1024"

echo ""
echo "=== Done. Model at results/IR_Module/ir_module_WR50_L2-3_PS3_1024_colorjitter_p0.1/ ==="
