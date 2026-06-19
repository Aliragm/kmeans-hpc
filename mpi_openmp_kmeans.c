#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <omp.h>
#include <mpi.h>
#include "dataset.h"

void mpi_openmp_kmeans(
    int k,
    float* flat_dataset,
    int row_count,
    const char* original_filename,
    int start_row,
    int start_col,
    int end_col
) {
    if (row_count <= 0 || flat_dataset == NULL) {
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
        generate_centroids(k, flat_dataset, flat_centroids, row_count, num_cols);
    }

    MPI_Bcast(flat_centroids, k * num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 1) {
        printf("Processo %d recebeu os centróides:\n", rank);
        for (int c = 0; c < k; c++) {
            printf("Centróide %d: ", c);
            for (int j = 0; j < num_cols; j++) {
                printf("%.2f ", flat_centroids[c * num_cols + j]);
            }
            printf("\n");
        }

        printf("Processo %d recebeu os dados:\n", rank);
        for (int i = 0; i < local_row_count; i++) {
            printf("Linha %d: ", i);
            for (int j = 0; j < num_cols; j++) {
                printf("%.2f ", local_dataset[i * num_cols + j]);
            }
            printf("\n");
        }
    }

    // TODO: implementar cálculo dos cluster e atualização dos centróides com OpenMP

    free(sendcounts_floats);
    free(displs_floats);
    free(sendcounts_ints);
    free(displs_ints);
    free(local_dataset);
    free(local_assignments);
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5) {
        if (rank == 0) {
            printf("Uso: %s <k> <arquivo_dataset> <start_row> <start_col> <end_col>\n", argv[0]);
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

    float* dataset = read_csv_to_floats(filename, &row_count, start_col, end_col, 1);

    if (dataset == NULL) {
        printf("Error at loading the dataset.\n");
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

    MPI_Finalize();
    return 0;
}