#!/bin/bash
#SBATCH --job-name=log-linear-real-sz
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -p Adelie

source ~/start-pyenv
source DisclosureRisk/tf-cpu/bin/activate

python DisclosureRisk/Experiments/log-linear-model/log_linear_real.py
