#!/bin/bash

#SBATCH --job-name=partis_oacc_uni_mem
#SBATCH --output=out_partis_oacc_uni_mem_%j.out
#SBATCH --error=out_partis_oacc_uni_mem_%j.err
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1

make >> make.out || exit 1
./partis_oacc_uni_mem 1000 1
./partis_oacc_uni_mem 1000 0
./partis_oacc_uni_mem 1500 0
./partis_oacc_uni_mem 2000 0
