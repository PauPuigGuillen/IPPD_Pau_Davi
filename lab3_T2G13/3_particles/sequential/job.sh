#!/bin/bash

#SBATCH --job-name=partis_seq
#SBATCH --output=out_partis_seq_%j.out
#SBATCH --error=out_partis_seq_%j.err
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1

make >> make.out || exit 1
./partis_seq 1000 1