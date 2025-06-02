#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH -a 1-3:1
#$ -N SZ_NY_mc

source /etc/profile

module add opence/1.10.0

python DisclosureRisk/Experiments/SZ_NY/BNP_mc/SZ_NY_tau.py
