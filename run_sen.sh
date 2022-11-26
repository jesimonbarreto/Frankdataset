#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --qos=Medium
#SBATCH --mem=230GB 
#SBATCH -o ./outfiles/%J.out


conda remove --name frankdataset --all 
conda env create -f environment.yml
source activate frankdataset 
conda info --envs 

srun python ./main_orig.py ./data PAMA-USC Sen-ss