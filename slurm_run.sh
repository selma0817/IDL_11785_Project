#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --mail-user=YIP33@pitt.edu

#SBATCH --output=/ihome/hkarim/yip33/IDL_11785_Project/run_logs/%x_%j.out
#SBATCH --error=/ihome/hkarim/yip33/IDL_11785_Project/run_logs/%x_%j.err



echo "project run"
module load gcc/9.2.0
# Source conda setup
eval "$(conda shell.bash hook)"

source /ix1/hkarim/yip33/custom_miniconda/bin/activate cvt_env

nvidia-smi
echo "rcvt srun"
srun --cpu-bind=none python /ihome/hkarim/yip33/IDL_11785_Project/train.py --model rcvt --resume /ix1/hkarim/yip33/IDL_11785_project/checkpoints/rcvt/baseline/model_last.pth
echo "finish rcvt run"

