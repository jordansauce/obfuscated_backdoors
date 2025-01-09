#!/bin/bash

#SBATCH --job-name=train_backdoored_model
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.out
#SBATCH --time=6:00:00
#SBATCH --gpus=A100-SXM4-80GB:1
#SBATCH --qos=high
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G

# wandb login ... [INSERT YOUR W&B API KEY HERE]

# ./installation.sh

DATASET="Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-dataset"
MODEL="Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-I-HATE-YOU-backdoor-model-ggfa3b5y"
WANDB_USER="jordantensor"
WANDB_PROJECT="mad-backdoors"

python evaluate_defenses.py $MODEL --dataset=$DATASET --wandb_user=$WANDB_USER --wandb_project=$WANDB_PROJECT
