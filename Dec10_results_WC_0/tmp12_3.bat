#!/bin/bash

#SBATCH -J RNA3_wl

#SBATCH -o out-%j

#SBATCH -e err-%j

#SBATCH -N 1 #number of nodes

#SBATCH -n 16 #total number of cores

#SBATCH -t 240:00:00

# #SBATCH --exclude=c1-[1-4]

# module load intel/17.0.4.196
module swap gnu  intel/17.0.4.196
module load mvapich2/2.2
module load ohpc-intel-mvapich2

python tmp12_3.py

