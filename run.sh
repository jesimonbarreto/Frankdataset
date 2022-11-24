#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem=230GB 
#SBATCH -o ./outfiles/%J.out
#SBATCH --nodelist=viper01


conda remove --name frankdataset --all 
conda env create -f environment.yml
source activate frankdataset 
conda info --envs 

srun python ./main_orig.py