#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --job-name=mocov3_convnext_no_cfm_v1
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vjh9526@nyu.edu
#SBATCH --output=/scratch/vjh9526/dl-fall-2025/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/vjh9526/dl-fall-2025/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --requeue

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec --bind /scratch $nv --overlay /scratch/vjh9526/dl-fall-2025/ssl_env/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/vjh9526/dl-fall-2025/code
export WANDB_API_KEY=2934cbfb64322bfaaf1aaa6657e1e9774a28fc8a
torchrun --nproc_per_node=2 scripts/train.py --data_dir /scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/train --output_dir /scratch/vjh9526/dl-fall-2025/outputs/mocov3_convnext_no_cfm --method moco --backbone convnext_tiny --batch_size 256 --run_name mocov3_convnext_no_cfm_v1
exit
"