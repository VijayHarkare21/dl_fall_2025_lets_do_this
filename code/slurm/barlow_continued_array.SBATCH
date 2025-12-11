#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --job-name=new_pretraining_tests_continued
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vjh9526@nyu.edu
#SBATCH --output=/scratch/vjh9526/dl-fall-2025/slurm_logs/%j_%x.out
#SBATCH --error=/scratch/vjh9526/dl-fall-2025/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

CONFIG=$(sed -n "1p" /scratch/vjh9526/dl-fall-2025/code/slurm/config_remaining.txt)
read EPOCHS WARMUP_EPOCHS METHOD ARCH BATCH_SIZE WANDB_RUN_NAME <<< $CONFIG

OUTDIR="/scratch/vjh9526/dl-fall-2025/code/new_checkpoints/${WANDB_RUN_NAME}/"

singularity exec --bind /scratch $nv --overlay /scratch/vjh9526/dl-fall-2025/ssl_env/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/vjh9526/dl-fall-2025/code
export WANDB_API_KEY=2934cbfb64322bfaaf1aaa6657e1e9774a28fc8a
torchrun --nproc_per_node=2 new_training_pipeline.py --method $METHOD --arch $ARCH --batch_size $BATCH_SIZE --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS --wandb_run_name $WANDB_RUN_NAME --output_dir "$OUTDIR"
exit
"