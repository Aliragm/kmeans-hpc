#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "dataset.h"

// converter matriz de strings para array 1d contiguo de floats
float* flatten_dataset(char*** dataset_original, int row_count, int num_cols, int start_col) {
    float* flat_data = malloc(row_count * num_cols * sizeof(float));
    for (int i = 0; i < row_count; i++) {
        for (int j = 0; j < num_cols; j++) {
            flat_data[i * num_cols + j] = atof(dataset_original[i][j + start_col]);
        }
    }
    return flat_data;
}

// converter matriz de centroides para array 1d contiguo
float* flatten_centroids(float** centroids_originais, int k, int num_cols) {
    float* flat_centroids = malloc(k * num_cols * sizeof(float));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < num_cols; j++) {
            flat_centroids[i * num_cols + j] = centroids_originais[i][j];
        }
    }
    return flat_centroids;
}

void sequential_kmeans(int k, char*** dataset_original, int row_count, const char* original_filename, int start_row, int start_col, int end_col){
    if (row_count <= 0 || dataset_original == NULL) return;

    int num_cols = (end_col - start_col) + 1;

    // achatar dados antes de iniciar o cronometro (otimizacao de cpu)
    float* flat_dataset = flatten_dataset(dataset_original, row_count, num_cols, start_col);
    
    float** centroids_temp = malloc(k * sizeof(float*));
    generate_centroids(k, dataset_original, centroids_temp, row_count, start_col, end_col);
    float* flat_centroids = flatten_centroids(centroids_temp, k, num_cols);

    int* assignments = malloc(row_count * sizeof(int));
    for (int i = 0; i < row_count; i++) assignments[i] = -1;

    clock_t tempo_inicio = clock();

    int converged = 0;
    int iterations = 0;

    while (converged == 0) {
        int changed = 0;
        
        // 1. assimilar aos centroides
        for (int i = 0; i < row_count; i++) {
            float min_dist = -1.0;
            int best_cluster = -1;
            
            for (int c = 0; c < k; c++) {
                float dist = 0.0;
                for (int j = 0; j < num_cols; j++) {
                    float diff = flat_dataset[i * num_cols + j] - flat_centroids[c * num_cols + j];
                    dist += diff * diff;
                }
                
                if (min_dist == -1.0 || dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            if (best_cluster != assignments[i]) {
                assignments[i] = best_cluster;
                changed = 1;
            }
        } 

        // 2. atualizar centroides
        if (changed == 1) {
            int* counts = calloc(k, sizeof(int));
            
            // zerar centroides atuais
            for(int c = 0; c < k; c++) {
                for(int j = 0; j < num_cols; j++) flat_centroids[c * num_cols + j] = 0.0;
            }
            
            // somar posicoes dos pontos
            for(int i = 0; i < row_count; i++) {
                int cluster = assignments[i];
                counts[cluster]++;
                for(int j = 0; j < num_cols; j++) {
                    flat_centroids[cluster * num_cols + j] += flat_dataset[i * num_cols + j];
                }
            }
            
            // calcular media
            for(int c = 0; c < k; c++) {
                if(counts[c] > 0) {
                    for(int j = 0; j < num_cols; j++) flat_centroids[c * num_cols + j] /= counts[c];
                }
            }
            free(counts);
        } else {
            converged = 1;
            printf("Convergiu apos %d iteracoes\n", iterations);
        }

        iterations++;
        if (iterations > 300) break;
    }

    clock_t tempo_fim = clock();
    double tempo_gasto = (double)(tempo_fim - tempo_inicio) / CLOCKS_PER_SEC;
    
    printf("\n--- metricas do baseline otimizado ---\n");
    printf("tempo de execucao: %f segundos\n", tempo_gasto);

    export_dataset_with_clusters(original_filename, "clusters_sequential_kmeans_otimizado.csv", assignments, row_count, start_row);

    free(flat_dataset);
    free(flat_centroids);
    for(int c = 0; c < k; c++) free(centroids_temp[c]);
    free(centroids_temp);
    free(assignments);
}

int main(int argc, char *argv[]){
    srand(time(NULL));

    int row_count = 0;
    const char* filename = "Iris.csv";
    
    char*** dataset = read_csv(filename, &row_count, 1);

    if (dataset == NULL) {
        printf("Error at loading the dataset.\n");
        return 1;
    }

    int start_row = 1; 
    int start_col = 1;
    int end_col = 4;

    sequential_kmeans(3, dataset, row_count, filename, start_row, start_col, end_col);

    free_dataset(dataset, row_count);

    return 0;
}
