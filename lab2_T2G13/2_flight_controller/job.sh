#!/bin/bash

#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_cholesky_%j.out
#SBATCH --error=out_cholesky_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=192
#SBATCH --nodes=1
#SBATCH --time=00:05:00

make >> make.out || exit 1      # Exit if make fails

./cholesky 3000

