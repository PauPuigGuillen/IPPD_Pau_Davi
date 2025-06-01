#!/bin/bash

#SBATCH --job-name=vecadd_oacc
#SBATCH --output=out_vecadd_oacc_%j.out
#SBATCH --error=out_vecadd_oacc_%j.err
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1

make >> make.out || exit 1
./vecadd_oacc 500