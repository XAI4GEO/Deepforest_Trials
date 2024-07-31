#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=genoa

source ~/.bashrc
conda activate jupyter_dask

python /home/oku2/Model_Trials/siamese_network/reforestree_netflora/training_v5.py
