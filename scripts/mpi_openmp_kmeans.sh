#!/bin/bash
#SBATCH --job-name=mpi_openmp_kmeans
#SBATCH --output=mpi_openmp_kmeans_%j.out
#SBATCH --error=mpi_openmp_kmeans_%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-0:10

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# <k>: clusters
# <arquivo_dataset>: dataset CSV
# <start_row>: linha inicial para leitura do dataset (0-indexed)
# <start_col>: coluna inicial para leitura do dataset (0-indexed)
# <end_col>: coluna final para leitura do dataset (0-indexed, inclusive)
mpirun ./mpi_openmp_kmeans 3 ${1:-Iris.csv} 1 1 4
