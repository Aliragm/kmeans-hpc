#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <omp.h>
#include "dataset.h"

#define MAX_ITERATIONS 300

void openmp_gpu_kmeans(
    int k,
    float* flat_dataset,
    int row_count,
    const char* original_filename,
    int start_row,
    int start_col,
    int end_col
) {
    if (row_count <= 0 || flat_dataset == NULL) return;

    int num_cols = (end_col - start_col) + 1;

    float* flat_centroids = malloc(k * num_cols * sizeof(float));
    generate_centroids(k, flat_dataset, flat_centroids, row_count, num_cols);

    int* assignments = malloc(row_count * sizeof(int));
    for (int i = 0; i < row_count; i++) assignments[i] = -1;

    // buffers auxiliares usados so na gpu para recalculo dos centroides
    float* centroid_sums = calloc(k * num_cols, sizeof(float));
    int* cluster_counts = calloc(k, sizeof(int));

    int converged = 0;
    int iterations = 0;

    double start_time = omp_get_wtime();

    // mantem os dados na gpu durante todas as iteracoes para evitar
    // transferencias repetidas. dataset: so leitura (to). centroides e
    // assignments: leitura/escrita (tofrom). buffers auxiliares: so gpu (alloc)
    #pragma omp target data \
        map(to: flat_dataset[0:row_count * num_cols]) \
        map(tofrom: flat_centroids[0:k * num_cols], assignments[0:row_count]) \
        map(alloc: centroid_sums[0:k * num_cols], cluster_counts[0:k])
    {
        while (converged == 0 && iterations < MAX_ITERATIONS) {
            int changed = 0;

            // fase 1: atribuicao. teams distribute divide as iteracoes entre
            // os blocos da gpu, parallel for paraleliza dentro de cada bloco.
            // reduction acumula o flag de mudanca sem condicao de corrida
            #pragma omp target teams distribute parallel for \
                reduction(+:changed)
            for (int i = 0; i < row_count; i++) {
                float min_dist = FLT_MAX;
                int best_cluster = -1;

                for (int c = 0; c < k; c++) {
                    float dist = 0.0f;
                    for (int j = 0; j < num_cols; j++) {
                        float diff = flat_dataset[i * num_cols + j] - flat_centroids[c * num_cols + j];
                        dist += diff * diff;
                    }

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }

                if (best_cluster != assignments[i]) {
                    assignments[i] = best_cluster;
                    changed++;
                }
            }

            if (changed > 0) {
                // zera os acumuladores na gpu antes de recalcular
                #pragma omp target teams distribute parallel for
                for (int i = 0; i < k * num_cols; i++) {
                    centroid_sums[i] = 0.0f;
                    if (i < k) cluster_counts[i] = 0;
                }

                // fase 2: acumulacao. atomic necessario pois threads diferentes
                // podem escrever no mesmo cluster simultaneamente
                #pragma omp target teams distribute parallel for
                for (int i = 0; i < row_count; i++) {
                    int cluster = assignments[i];
                    #pragma omp atomic
                    cluster_counts[cluster]++;
                    for (int j = 0; j < num_cols; j++) {
                        #pragma omp atomic
                        centroid_sums[cluster * num_cols + j] += flat_dataset[i * num_cols + j];
                    }
                }

                // fase 3: recalculo dos centroides como media dos pontos do cluster
                #pragma omp target teams distribute parallel for
                for (int c = 0; c < k; c++) {
                    if (cluster_counts[c] > 0) {
                        for (int j = 0; j < num_cols; j++) {
                            flat_centroids[c * num_cols + j] =
                                centroid_sums[c * num_cols + j] / (float)cluster_counts[c];
                        }
                    }
                }
            } else {
                converged = 1;
                printf("Convergiu apos %d iteracoes\n", iterations);
            }

            iterations++;
        }
    }

    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;

    printf("\n--- metricas do openmp gpu kmeans ---\n");
    printf("Arquivo de entrada: %s\n", original_filename);
    printf("Numero de clusters (k): %d\n", k);
    printf("Numero de iteracoes: %d\n", iterations);
    printf("Tempo total de execucao: %f segundos\n", total_time);

    export_dataset_with_clusters(
        original_filename,
        "clusters_openmp_gpu_kmeans.csv",
        assignments,
        row_count,
        start_row
    );

    free(flat_centroids);
    free(assignments);
    free(centroid_sums);
    free(cluster_counts);
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        printf("Uso: %s <k> <arquivo_dataset> <start_row> <start_col> <end_col>\n", argv[0]);
        return 1;
    }

    int k = atoi(argv[1]);
    const char* filename = argv[2];
    int start_row = atoi(argv[3]);
    int start_col = atoi(argv[4]);
    int end_col = atoi(argv[5]);

    int row_count = 0;
    srand(time(NULL));

    float* dataset = read_csv_to_floats(filename, &row_count, start_col, end_col, 1);

    if (dataset == NULL || row_count <= 0) {
        printf("Erro ao ler o dataset ou dataset vazio.\n");
        return 1;
    }

    openmp_gpu_kmeans(k, dataset, row_count, filename, start_row, start_col, end_col);

    free(dataset);
    return 0;
}
