#!/bin/bash

#SBATCH --job-name=partis_oacc_async
#SBATCH --output=out_partis_oacc_async_%j.out
#SBATCH --error=out_partis_oacc_async_%j.err
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1

make >> make.out || exit 1
./partis_oacc_async 1000 0
nsys profile ./partis_oacc_async 1000 0
