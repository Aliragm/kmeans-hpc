#!/bin/bash
#SBATCH --job-name=cuda_kmeans
#SBATCH --output=cuda_kmeans_%j.out
#SBATCH --error=cuda_kmeans_%j.err
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00

ulimit -s unlimited
module load compilers/nvidia/nvhpc/24.11

# dataset.c e compilado como C (linkagem C) e linkado com o kernel CUDA.
# -arch=sm_70 corresponde a GPU V100 da particao gpu-8-v100
nvcc -O3 -arch=sm_70 -c cuda_kmeans.cu -o cuda_kmeans.o
gcc -O3 -c dataset.c -o dataset.o
nvcc -O3 cuda_kmeans.o dataset.o -o cuda_kmeans -lm

# <k>: clusters
# <arquivo_dataset>: dataset CSV
# <start_row>: linha inicial para leitura do dataset (0-indexed)
# <start_col>: coluna inicial para leitura do dataset (0-indexed)
# <end_col>: coluna final para leitura do dataset (0-indexed, inclusive)
./cuda_kmeans 3 Iris.csv 1 1 4
