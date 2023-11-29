#!/bin/bash
#SBATCH --job-name=test_mario_smaller
#SBATCH --time=360:00
#SBATCH -p gpu
#SBATCH -G 8
#SBATCH -c 8
#SBATCH --mem=8GB

# load all of the needed packages
module load system openssl
module load python/3.9.0
ml load math
ml load torch
ml load gcc
pip3 install -r requirements.txt

# run the file
python3 baseline.py