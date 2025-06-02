#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH -a 3-3:1
#$ -N SZ_NY_mc

source /etc/profile

module add opence/1.10.0
module add cuda/11.8

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/shared_apps/packages/cuda-11.8/nvvm/libdevice"

python DisclosureRisk/experiments/SZ_NY/SZ_BNP_mc/SZ_NY_tau.py
