#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --job-name=pcb_m4_512
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

M4_512="results/IR_Module/ir_module_WR50_L2-3_PS3_512_m4_p0.1/models/mvtec_ir_module"

echo "=== Model4 512: good (ws + ns boards) ==="
python yolo/640C/infer_pcb.py \
  --image_dir data/ir_module_512/ir_module/test/good \
  --patchcore_path "$M4_512" \
  --resize 512 --cropsize 512 \
  --save

echo ""
echo "=== Model4 512: defect (defect-01 to 13) ==="
python yolo/640C/infer_pcb.py \
  --image_dir data/ir_module_512/ir_module/test/defect \
  --patchcore_path "$M4_512" \
  --resize 512 --cropsize 512 \
  --save

echo ""
echo "=== Model4 512: defect_type1 — 10px dilation for background screws ==="
python yolo/640C/infer_pcb.py \
  --image_dir data/ir_module_512/ir_module/test/defect_type1 \
  --patchcore_path "$M4_512" \
  --resize 512 --cropsize 512 \
  --suppress_dilation 10 \
  --save

echo ""
echo "=== Model4 512: defect_type2 — 10px dilation for background screws ==="
python yolo/640C/infer_pcb.py \
  --image_dir data/ir_module_512/ir_module/test/defect_type2 \
  --patchcore_path "$M4_512" \
  --resize 512 --cropsize 512 \
  --suppress_dilation 10 \
  --save

echo ""
echo "=== Done. Results at results/yolo/pcb_inspection/ir_module_WR50_L2-3_PS3_512_m4_p0.1/ ==="
