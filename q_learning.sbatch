#!/bin/bash
#SBATCH --job-name=test_mario
#SBATCH --time=180:00
#SBATCH -p gpu
#SBATCH -G 80
#SBATCH -c 320
#SBATCH --mem=24GB

# load all of the needed packages
module load system openssl
module load python/3.9.0
ml load math
ml load torch
ml load gcc
pip3 install -r requirements.txt

# run the file
python3 baseline.py