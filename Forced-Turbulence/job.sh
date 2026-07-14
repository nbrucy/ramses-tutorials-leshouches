#!/bin/bash
#
#SBATCH --job-name=yourname_p7
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --partition=Cascade-GPU
#SBATCH --reservation=Houches2026
#SBATCH --mem-per-cpu=3G

mpirun ./ramses3d driving.nml > out.${SLURM_JOB_ID}
