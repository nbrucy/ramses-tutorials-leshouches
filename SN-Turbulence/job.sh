#!/bin/bash
#
#SBATCH --job-name=brucy_p3
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --partition=Cascade
##SBATCH --reservation=Houches2026

mpirun ./ramses3d run.nml > out.${SLURM_JOB_ID}
