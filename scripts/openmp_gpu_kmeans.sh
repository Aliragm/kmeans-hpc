#!/bin/bash
#SBATCH --job-name=openmp_gpu_kmeans
#SBATCH --output=openmp_gpu_kmeans_%j.out
#SBATCH --error=openmp_gpu_kmeans_%j.err
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00

ulimit -s unlimited
module load compilers/nvidia/nvhpc/24.11

nvc -mp=gpu -Minfo=mp -O2 -o openmp_gpu_kmeans openmp_gpu_kmeans.c dataset.c

# OMP_TARGET_OFFLOAD=MANDATORY garante que o codigo rode na GPU,
# abortando caso o offload nao seja possivel
# <k>: clusters
# <arquivo_dataset>: dataset CSV
# <start_row>: linha inicial para leitura do dataset (0-indexed)
# <start_col>: coluna inicial para leitura do dataset (0-indexed)
# <end_col>: coluna final para leitura do dataset (0-indexed, inclusive)
OMP_TARGET_OFFLOAD=MANDATORY ./openmp_gpu_kmeans 3 Iris.csv 1 1 4
