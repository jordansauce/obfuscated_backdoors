#!/bin/bash

#SBATCH --job-name=train_backdoored_model
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.out
#SBATCH --time=6:00:00
#SBATCH --gpus=A100-SXM4-80GB:1
#SBATCH --qos=high
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=16G

uv run train_backdoored_model_no_explicit_obfuscation.py
