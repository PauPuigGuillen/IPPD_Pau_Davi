#!/bin/bash

#SBATCH --job-name=partis_oacc_prog_managed
#SBATCH --output=out_partis_oacc_prog_managed_%j.out
#SBATCH --error=out_partis_oacc_prog_managed_%j.err
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1

make >> make.out || exit 1
nsys profile ./partis_oacc_prog_managed 1000 0
