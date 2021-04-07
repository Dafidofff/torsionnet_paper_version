#!/bin/bash
#
#SBATCH --job-name=gfnxtb
#SBATCH -n 1
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks=16
#SBATCH -p ivy

#./generate_conformers.py cyclohexane > cyclohexane.out
#./generate_conformers.py n_butane > n_butane.out
./generate_conformers.py cyclohexane_test 10000 > cyclohexane_test.out
