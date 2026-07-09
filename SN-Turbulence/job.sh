#!/bin/bash
#
#SBATCH --job-name=yourname_p3
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --partition=Cascade-GPU
#SBATCH --reservation=Houches2026

mpirun ./ramses3d run.nml > out.${SLURM_JOB_ID}
