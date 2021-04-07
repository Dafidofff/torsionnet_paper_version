#!/bin/bash
#
#SBATCH --job-name=gfnxtb
#SBATCH -n 1
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks=16
#SBATCH -p ivy

./enumerate.py > enumerate.out
