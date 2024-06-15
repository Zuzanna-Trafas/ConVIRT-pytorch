#!/bin/bash
#SBATCH -J pretrain-convirt
#SBATCH -N 1
#SBATCH -p lrz-v100x2
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=7-12:00:00
#SBATCH -o %x.%j.%N.out
#SBATCH -e %x.%j.%N.err

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate ConVIRT # activate your environment

cd ~/master_thesis/chest-xray/baselines/ConVIRT-pytorch

srun python run.py