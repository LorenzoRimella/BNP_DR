#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH -a 3-3:1
#$ -N SZ_NY_mc

source /etc/profile

module add cuda/11.2
module add anaconda3

source activate tf-gpu

python DisclosureRisk/Experiments/SZ_NY/SZ_BNP_mc/SZ_NY_tau.py
