#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH -a 1-3:1
#$ -N repeated MCMC


source /etc/profile

module add anaconda3/2023.09
module add cuda/12.5

source activate tf-gpu

python DisclosureRisk/Experiments/MCMC_repeated/SZ_tau.py
