#!/bin/bash
#SBATCH --job-name=repeated MCMC
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH -p PenguinPartition
#SBATCH -a 1-3:1

source ~/start-pyenv
source DisclosureRisk/tf-cpu/bin/activate

python DisclosureRisk/experiments/Synthetic/BNP_mc_repeated/synth_tau.py
