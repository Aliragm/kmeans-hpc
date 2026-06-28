#!/bin/bash
#SBATCH --job-name=sequential_kmeans
#SBATCH --output=sequential_kmeans_%j.out
#SBATCH --error=sequential_kmeans_%j.err
#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:10

# Executa o baseline sequencial otimizado (aceita dataset como parametro, padrao Iris.csv)
./kmeans_otimizado 3 ${1:-Iris.csv} 1 1 4
