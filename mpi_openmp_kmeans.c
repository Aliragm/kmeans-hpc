#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <omp.h>
#include <mpi.h>
#include "dataset.h"

#define MAX_ITERATIONS 300

void mpi_openmp_kmeans(
    int k,
    float* flat_dataset,
    int row_count,
    const char* original_filename,
    int start_row,
    int start_col,
    int end_col
) {
    if (row_count <= 0) {
        return;
    }

    int num_cols = (end_col - start_col) + 1;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Vetores de contagem e deslocamento dos dados do dataset e do índices dos clusters
    int* sendcounts_floats = malloc(size * sizeof(int));
    int* displs_floats = malloc(size * sizeof(int));
    int* sendcounts_ints = malloc(size * sizeof(int));
    int* displs_ints = malloc(size * sizeof(int));

    int rows_per_proc = row_count / size;
    int remaining_rows = row_count % size;

    int current_displ_floats = 0;
    int current_displ_ints = 0;

    for (int i = 0; i < size; i++) {
        // Caso a divisão do número de linhas por processo não seja exata, os primeiros
        // processos recebem uma linha a mais de modo que os datapoints restantes
        // sejam distribuídas entre os processos de forma justa.
        int rows_for_process_i = rows_per_proc + (i < remaining_rows ? 1 : 0);

        sendcounts_floats[i] = rows_for_process_i * num_cols;
        displs_floats[i] = current_displ_floats;
        current_displ_floats += sendcounts_floats[i];

        sendcounts_ints[i] = rows_for_process_i;
        displs_ints[i] = current_displ_ints;
        current_displ_ints += sendcounts_ints[i];
    }

    int local_row_count = sendcounts_ints[rank];
    float* local_dataset = malloc(local_row_count * num_cols * sizeof(float));
    int* local_assignments = malloc(local_row_count * sizeof(int));

    for (int i = 0; i < local_row_count; i++) {
        local_assignments[i] = -1;
    }

    // Envio das porções do dataset para cada processo
    MPI_Scatterv(
        flat_dataset,
        sendcounts_floats,
        displs_floats,
        MPI_FLOAT,
        local_dataset,
        sendcounts_floats[rank],
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    // Todos os processos terão o mesmo vetor de centróides, que será atualizado a cada
    // iteração, porém somente o processo 0 que irá inicializá-lo e depois disso ele
    // irá compartilhar esse vetor com todos os outros processos.
    float* flat_centroids = malloc(k * num_cols * sizeof(float));
    if (rank == 0) {
        srand(time(NULL));
        generate_centroids(k, flat_dataset, flat_centroids, row_count, num_cols);
    }
    MPI_Bcast(flat_centroids, k * num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int converged = 0;
    int iterations = 0;

    float* local_centroid_sums = malloc(k * num_cols * sizeof(float));
    int* local_cluster_counts = malloc(k * sizeof(int));

    float* global_centroid_sums = calloc(k * num_cols, sizeof(float));
    int* global_cluster_counts = calloc(k, sizeof(int));

    double start_time = MPI_Wtime();

    while (converged == 0 && iterations < MAX_ITERATIONS) {
        int local_changed = 0;

        for (int i = 0; i < k * num_cols; i++) {
            local_centroid_sums[i] = 0.0f;
        }
        for (int i = 0; i < k; i++) {
            local_cluster_counts[i] = 0;
        }

        #pragma omp parallel
        {
            float* thread_centroid_sums = calloc(k * num_cols, sizeof(float));
            int* thread_cluster_counts = calloc(k, sizeof(int));
            int thread_changed = 0;

            #pragma omp for
            for (int i = 0; i < local_row_count; i++) {
                // Cada thread vai calcular a distância de um datapoint para cada
                // centróide e determinal a qual cluster ele pertence
                float min_dist = -1.0f;
                int best_cluster = -1;

                for (int c = 0; c < k; c++) {
                    float dist = 0.0f;
                    for (int j = 0; j < num_cols; j++) {
                        float diff = local_dataset[i * num_cols + j] - flat_centroids[c * num_cols + j];
                        dist += diff * diff;
                    }

                    if (min_dist == -1.0f || dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }

                if (best_cluster != local_assignments[i]) {
                    local_assignments[i] = best_cluster;
                    thread_changed = 1;
                }

                // Após determinar a qual cluster o datapoint pertence, cada thread vai
                // atualizar os somatórios das features dos centróides e também da
                // contagem de elementos em cada cluster
                thread_cluster_counts[best_cluster]++;
                for (int j = 0; j < num_cols; j++) {
                    thread_centroid_sums[best_cluster * num_cols + j] += local_dataset[i * num_cols + j];
                }
            }

            // Como cada thread está contando o número de elementos em cada cluster e
            // realizando o somatório das features dos centróides, é necessário que após
            // a finalização do trabalho de cada thread, os resultados obtidos sejam
            // somandos aos resultados do processo. Para isso, é necessário que essa
            // operação seja realizada em uma região crítica para que não haja condições
            // de corrida entre as threads e os resultados sejam consistentes.
            #pragma omp critical
            {
                if (thread_changed) {
                    local_changed = 1;
                }
                for (int c = 0; c < k; c++) {
                    local_cluster_counts[c] += thread_cluster_counts[c];
                    for (int j = 0; j < num_cols; j++) {
                        local_centroid_sums[c * num_cols + j] += thread_centroid_sums[c * num_cols + j];
                    }
                }
            }
            free(thread_centroid_sums);
            free(thread_cluster_counts);
        }

        // Redução para verificar se algum processo teve mudanças em seus clusters
        int global_changed = 0;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        MPI_Allreduce(local_cluster_counts, global_cluster_counts, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_centroid_sums, global_centroid_sums, k * num_cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // Atualização dos centróides com base nos somatórios globais e nas contagens globais
        if (global_changed == 1) {
            for (int c = 0; c < k; c++) {
                if (global_cluster_counts[c] > 0) {
                    for (int j = 0; j < num_cols; j++) {
                        flat_centroids[c * num_cols + j] = (float) global_centroid_sums[c * num_cols + j] / (float) global_cluster_counts[c];
                    }
                }
            }
        } else {
            converged = 1;
            if (rank == 0) {
                printf("Convergiu apos %d iteracoes\n", iterations);
            }
        }

        iterations++;
    }

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    if (rank == 0) {
        printf("Cálculo do K-means paralelo com MPI e OpenMP.\n");
        printf("Número de processos: %d\n", size);
        printf("Número de threads por processo: %d\n", omp_get_max_threads());
        printf("Arquivo de entrada: %s\n", original_filename);
        printf("Número de clusters (k): %d\n", k);
        printf("Número de iterações: %d\n", iterations);
        printf("Tempo total de execução: %f segundos\n", total_time);
    }

    int* global_assignments = NULL;
    if (rank == 0) {
        global_assignments = malloc(row_count * sizeof(int));
    }

    MPI_Gatherv(
        local_assignments,
        sendcounts_ints[rank],
        MPI_INT,
        global_assignments,
        sendcounts_ints,
        displs_ints,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        export_dataset_with_clusters(
            original_filename,
            "clusters_mpi_openmp_kmeans.csv",
            global_assignments,
            row_count,
            start_row
        );
        free(global_assignments);
    }

    free(sendcounts_floats);
    free(displs_floats);
    free(sendcounts_ints);
    free(displs_ints);

    free(local_dataset);
    free(local_assignments);

    free(local_centroid_sums);
    free(local_cluster_counts);
    free(global_centroid_sums);
    free(global_cluster_counts);
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) {
            printf(
                "Uso: %s <k> <arquivo_dataset> <start_row> <start_col> <end_col>\n",
                argv[0]
            );
        }
        MPI_Finalize();
        return 1;
    }

    int k = atoi(argv[1]);
    const char* filename = argv[2];
    int start_row = atoi(argv[3]);
    int start_col = atoi(argv[4]);
    int end_col = atoi(argv[5]);

    int row_count = 0;
    float* dataset = NULL;

    // Somente o processo 0 irá ler o dataset do arquivo CSV e alocar na memória
    if (rank == 0) {
        dataset = read_csv_to_floats(filename, &row_count, start_col, end_col, 1);
        if (dataset == NULL) {
            row_count = 0;
        }
    }

    MPI_Bcast(&row_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (row_count <= 0) {
        if (rank == 0) {
            printf("Erro ao ler o dataset ou dataset vazio.\n");
        }
        MPI_Finalize();
        return 1;
    }

    mpi_openmp_kmeans(
        k,
        dataset,
        row_count,
        filename,
        start_row,
        start_col,
        end_col
    );

    if (rank == 0) {
        free(dataset);
    }

    MPI_Finalize();
    return 0;
}
