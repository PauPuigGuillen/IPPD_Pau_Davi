#!/bin/bash

#SBATCH --job-name=matmul
#SBATCH --output=out_matmul_%j.out
#SBATCH --error=out_matmul_%j.err
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1

make >> make.out || exit 1

echo "-------------Running matrix multiplication with size 1024...-------------"
./matmul 1024 1
echo "-------------Running matrix multiplication with size 2048...-------------"
./matmul 2048 1
echo "-------------Running matrix multiplication with size 4096...-------------"
./matmul 4096 1
echo "-------------Running matrix multiplication with size 8192...-------------"
./matmul 8192 1