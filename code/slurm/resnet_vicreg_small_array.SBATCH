#!/bin/bash
#SBATCH --array=0-2
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=latest_test_single_gpu
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vjh9526@nyu.edu
#SBATCH --output=/scratch/vjh9526/dl-fall-2025/slurm_logs/%A_%a_%x.out
#SBATCH --error=/scratch/vjh9526/dl-fall-2025/slurm_logs/%A_%a_%x.err
#SBATCH --export=ALL
#SBATCH --time=19:00:00
#SBATCH --gres=gpu
#SBATCH --requeue

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

# Read line corresponding to current array task ID
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /scratch/vjh9526/dl-fall-2025/code/slurm/config_resnet_vicreg_single_gpu.txt)
read EPOCHS WARMUP_EPOCHS MAIN_MAX_SAMPLES WANDB_RUN_NAME TRAIN_DATA_TYPE <<< $CONFIG

OUTDIR="/scratch/vjh9526/dl-fall-2025/checkpoints_${WANDB_RUN_NAME}/"

singularity exec --bind /scratch $nv --overlay /scratch/vjh9526/dl-fall-2025/ssl_env/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/vjh9526/dl-fall-2025/code
export WANDB_API_KEY=2934cbfb64322bfaaf1aaa6657e1e9774a28fc8a
python resnet_vicreg_train.py --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS --main_max_samples $MAIN_MAX_SAMPLES --train_data_type $TRAIN_DATA_TYPE --wandb_run_name $WANDB_RUN_NAME --output_dir $OUTDIR
exit
"