#!/bin/bash

#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_fc_%j.out
#SBATCH --error=out_fc_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --time=00:05:00

make >> make.out || exit 1      # Exit if make fails

mpirun -n 2 ./fc_mpi input_planes_test.txt 25 1 0