#!/bin/bash

#SBATCH --job-name=vecadd_cuda
#SBATCH --output=out_vecadd_cuda_%j.out
#SBATCH --error=out_vecadd_cuda_%j.err
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1

make >> make.out || exit 1
./vecadd_cuda 500
./vecadd_cuda 5000
./vecadd_cuda 50000
./vecadd_cuda 500000
./vecadd_cuda 5000000
./vecadd_cuda 50000000
./vecadd_cuda 500000000