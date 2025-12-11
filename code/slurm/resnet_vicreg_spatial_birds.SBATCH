#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=latest_test_single_gpu_v2
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vjh9526@nyu.edu
#SBATCH --output=/scratch/vjh9526/dl-fall-2025/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/vjh9526/dl-fall-2025/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=20:00:00
#SBATCH --gres=gpu
#SBATCH --requeue

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

OUTDIR="/scratch/vjh9526/dl-fall-2025/checkpoints_resnet_vicreg_spatial_birds/"

singularity exec --bind /scratch $nv --overlay /scratch/vjh9526/dl-fall-2025/ssl_env/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/vjh9526/dl-fall-2025/code
export WANDB_API_KEY=2934cbfb64322bfaaf1aaa6657e1e9774a28fc8a
python resnet_vicreg_train_spatial.py --epochs 100 --warmup_epochs 10 --main_max_samples 0 --train_data_type birds --wandb_run_name resnet_vicreg_spatial_birds --output_dir $OUTDIR --batch_size 256 --base_lr 0.45
exit
"