#!/bin/bash

#SBATCH --job-name=vecadd_cuda
#SBATCH --output=out_vecadd_cuda_%j.out
#SBATCH --error=out_vecadd_cuda_%j.err
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1

make >> make.out || exit 1

nvcc -o vecadd_cuda vecadd_cuda.cu 500