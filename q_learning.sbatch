#!/bin/bash
#SBATCH --job-name=mario_cpu
#SBATCH --time=360:00
#SBATCH -c 128
#SBATCH --mem=16GB

# load all of the needed packages
module load system openssl
module load python/3.9.0
ml load math
ml load torch
ml load gcc
pip3 install -r requirements.txt

# run the file
python3 baseline.py