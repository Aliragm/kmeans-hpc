#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "dataset.h"

void sequential_kmeans(int k, float* flat_dataset, int row_count, const char* original_filename, int start_row, int start_col, int end_col) {
    if (row_count <= 0 || flat_dataset == NULL) return;

    int num_cols = (end_col - start_col) + 1;

    float* flat_centroids = malloc(k * num_cols * sizeof(float));
    generate_centroids(k, flat_dataset, flat_centroids, row_count, num_cols);

    int* assignments = malloc(row_count * sizeof(int));
    for (int i = 0; i < row_count; i++) assignments[i] = -1;

    struct timespec tempo_inicio, tempo_fim;
    clock_gettime(CLOCK_MONOTONIC, &tempo_inicio);

    int converged = 0;
    int iterations = 0;

    while (converged == 0) {
        int changed = 0;

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

        if (changed == 1) {
            int* counts = calloc(k, sizeof(int));

            for (int c = 0; c < k; c++) {
                for (int j = 0; j < num_cols; j++) flat_centroids[c * num_cols + j] = 0.0;
            }

            for (int i = 0; i < row_count; i++) {
                int cluster = assignments[i];
                counts[cluster]++;
                for (int j = 0; j < num_cols; j++) {
                    flat_centroids[cluster * num_cols + j] += flat_dataset[i * num_cols + j];
                }
            }

            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    for (int j = 0; j < num_cols; j++) flat_centroids[c * num_cols + j] /= counts[c];
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

    clock_gettime(CLOCK_MONOTONIC, &tempo_fim);
    double tempo_gasto = (tempo_fim.tv_sec - tempo_inicio.tv_sec) + (tempo_fim.tv_nsec - tempo_inicio.tv_nsec) / 1e9;

    printf("\n--- metricas do baseline otimizado ---\n");
    printf("tempo de execucao: %f segundos\n", tempo_gasto);

    export_dataset_with_clusters(original_filename, "clusters_sequential_kmeans_otimizado.csv", assignments, row_count, start_row);

    free(flat_centroids);
    free(assignments);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    int row_count = 0;
    const char* filename = "Iris.csv";
    int start_row = 1;
    int start_col = 1;
    int end_col = 4;

    float* dataset = read_csv_to_floats(filename, &row_count, start_col, end_col, 1);

    if (dataset == NULL) {
        printf("Error at loading the dataset.\n");
        return 1;
    }

    sequential_kmeans(3, dataset, row_count, filename, start_row, start_col, end_col);

    free(dataset);

    return 0;
}
