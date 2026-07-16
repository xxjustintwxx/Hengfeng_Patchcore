#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --job-name=pcb_m3
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:00:00
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

M3="results/IR_Module/ir_module_WR50_L2-3_PS3_1024_colorjitter_p0.1/models/mvtec_ir_module"

echo "=== Model3 (colorjitter, no rotation, p=0.1): defect (defect-01 to 13) ==="
python yolo/infer_pcb.py \
  --image_dir data/ir_module_1024/ir_module/test/defect \
  --patchcore_path "$M3" \
  --save

echo ""
echo "=== Model3: good ==="
python yolo/infer_pcb.py \
  --image_dir data/ir_module_1024/ir_module/test/good \
  --patchcore_path "$M3" \
  --save

echo ""
echo "=== Model3: defect_type1 (defect-14 to 18) — 20px screw dilation for background screws ==="
python yolo/infer_pcb.py \
  --image_dir data/ir_module_1024/ir_module/test/defect_type1 \
  --patchcore_path "$M3" \
  --suppress_dilation 20 \
  --save

echo ""
echo "=== Model3: defect_type2 (defect-19 to 26) — 20px screw dilation for background screws ==="
python yolo/infer_pcb.py \
  --image_dir data/ir_module_1024/ir_module/test/defect_type2 \
  --patchcore_path "$M3" \
  --suppress_dilation 20 \
  --save

echo ""
echo "=== Done. Results at results/yolo/pcb_inspection/ir_module_WR50_L2-3_PS3_1024_colorjitter_p0.1/ ==="
