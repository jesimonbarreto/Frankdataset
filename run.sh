#!/bin/bash
#SBATCH --qos=high
#SBATCH -o ./outfiles/%J.out
#SBATCH --mem=230GB 

conda remove --name frankdataset --all 
conda env create -f environment.yml
source activate frankdataset 
conda info --envs 

srun python ./main_orig.py 10