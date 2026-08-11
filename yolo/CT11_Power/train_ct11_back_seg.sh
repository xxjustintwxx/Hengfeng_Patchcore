#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --job-name=ct11_back_yolo
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
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
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

echo "=== YOLOv11-seg CT11 Back Training (small model + board class) ==="
echo "Classes : board(1), component(1), connector(1), resistor(61)"
echo "Node    : $(hostname)"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python  : $(which python)"
echo ""

python yolo/CT11/train_ct11_back_seg.py
