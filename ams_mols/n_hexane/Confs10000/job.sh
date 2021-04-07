#!/bin/bash
#
#SBATCH --job-name=gfnxtb
#SBATCH -n 1
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks=16
#SBATCH -p ivy

./generate_conformers.py . 10000 > n_hexane.out
