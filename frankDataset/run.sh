#!/bin/bash
#SBATCH --qos=cpu
#SBATCH -o ./../2-residuals/slurm/%J.out
#SBATCH --mem=230GB 

#conda remove --name frankdataset --all 
conda env create -f environment.yml
source activate frankdataset 
conda info --envs 


srun python ./main_orig.py 10
srun python ./main_orig.py 20
srun python ./main_orig.py 30
srun python ./main_orig.py 50
srun python ./main_orig.py 100