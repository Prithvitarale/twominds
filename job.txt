#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=56000  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 08:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load cuda/10
/modules/apps/cuda/10.1.243/samples/bin/x86_64/linux/release/deviceQuery
python3 ./interface.py
 
