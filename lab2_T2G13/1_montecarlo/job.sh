#!/bin/bash

#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_montecarlo_%j.out
#SBATCH --error=out_montecarlo_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --time=00:05:00

make >> make.out || exit 1      # Exit if make fails

mpirun -n 12 ./montecarlo 4 100000000 10

