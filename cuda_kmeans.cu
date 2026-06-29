#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

// dataset.c e compilado como C; envolvemos o header em extern "C" para que o
// nvcc (que compila .cu como C++) referencie os simbolos com linkagem C.
extern "C" {
#include "dataset.h"
}

#define MAX_ITERATIONS 300

// macro de verificacao de erros: aborta e informa arquivo/linha caso uma
// chamada da runtime CUDA falhe, evitando que erros passem despercebidos
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "Erro CUDA em %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// fase 1: atribuicao. cada thread cuida de um datapoint, calcula a distancia
// euclidiana ao quadrado para cada centroide e escolhe o cluster mais proximo.
// se a atribuicao mudou, sinaliza no flag global compartilhado (changed).
__global__ void assign_clusters(
    const float* dataset,
    const float* centroids,
    int* assignments,
    int row_count,
    int num_cols,
    int k,
    int* changed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;

    float min_dist = FLT_MAX;
    int best_cluster = -1;

    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        for (int j = 0; j < num_cols; j++) {
            float diff = dataset[i * num_cols + j] - centroids[c * num_cols + j];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    if (best_cluster != assignments[i]) {
        assignments[i] = best_cluster;
        // varias threads podem sinalizar mudanca ao mesmo tempo; atomicOr
        // garante a escrita consistente do flag sem condicao de corrida
        atomicOr(changed, 1);
    }
}

// fase 2: acumulacao. cada thread soma as features do seu datapoint nos
// acumuladores do cluster correspondente. atomicAdd e necessario pois threads
// distintas podem escrever no mesmo cluster simultaneamente.
__global__ void accumulate_centroids(
    const float* dataset,
    const int* assignments,
    float* centroid_sums,
    int* cluster_counts,
    int row_count,
    int num_cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;

    int cluster = assignments[i];
    atomicAdd(&cluster_counts[cluster], 1);
    for (int j = 0; j < num_cols; j++) {
        atomicAdd(&centroid_sums[cluster * num_cols + j], dataset[i * num_cols + j]);
    }
}

// fase 3: recalculo. cada thread cuida de um cluster e atualiza suas features
// como a media dos pontos atribuidos a ele. k costuma ser pequeno, entao uma
// thread por cluster e suficiente.
__global__ void update_centroids(
    float* centroids,
    const float* centroid_sums,
    const int* cluster_counts,
    int k,
    int num_cols
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    if (cluster_counts[c] > 0) {
        for (int j = 0; j < num_cols; j++) {
            centroids[c * num_cols + j] =
                centroid_sums[c * num_cols + j] / (float)cluster_counts[c];
        }
    }
}

void cuda_kmeans(
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

    // centroides iniciais sao gerados na CPU (mesma rotina das outras versoes)
    // para que a comparacao de resultados entre implementacoes seja justa
    float* flat_centroids = (float*)malloc(k * num_cols * sizeof(float));
    generate_centroids(k, flat_dataset, flat_centroids, row_count, num_cols);

    int* assignments = (int*)malloc(row_count * sizeof(int));
    for (int i = 0; i < row_count; i++) assignments[i] = -1;

    // alocacao dos buffers na GPU
    float *d_dataset, *d_centroids, *d_centroid_sums;
    int *d_assignments, *d_cluster_counts, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_dataset, row_count * num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, k * num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroid_sums, k * num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assignments, row_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_counts, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    // eventos para medir o overhead de transferencia CPU<->GPU isoladamente,
    // metrica importante para a analise do projeto
    cudaEvent_t ev_h2d_start, ev_h2d_stop, ev_d2h_start, ev_d2h_stop;
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));

    struct timespec tempo_inicio, tempo_fim;
    clock_gettime(CLOCK_MONOTONIC, &tempo_inicio);

    // transferencia inicial CPU -> GPU (dataset, centroides e assignments).
    // o dataset e enviado uma unica vez e permanece na GPU por todas as iteracoes
    CUDA_CHECK(cudaEventRecord(ev_h2d_start));
    CUDA_CHECK(cudaMemcpy(d_dataset, flat_dataset,
                          row_count * num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, flat_centroids,
                          k * num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_assignments, assignments,
                          row_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_h2d_stop));

    int threads = 256;
    int blocks_points = (row_count + threads - 1) / threads;
    int blocks_clusters = (k + threads - 1) / threads;

    int converged = 0;
    int iterations = 0;

    while (converged == 0 && iterations < MAX_ITERATIONS) {
        // zera o flag de mudanca na GPU antes da fase de atribuicao
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

        assign_clusters<<<blocks_points, threads>>>(
            d_dataset, d_centroids, d_assignments,
            row_count, num_cols, k, d_changed);
        CUDA_CHECK(cudaGetLastError());

        // traz apenas o flag de volta para decidir a convergencia na CPU
        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

        if (changed != 0) {
            // zera os acumuladores na GPU antes de recalcular os centroides
            CUDA_CHECK(cudaMemset(d_centroid_sums, 0, k * num_cols * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_cluster_counts, 0, k * sizeof(int)));

            accumulate_centroids<<<blocks_points, threads>>>(
                d_dataset, d_assignments, d_centroid_sums, d_cluster_counts,
                row_count, num_cols);
            CUDA_CHECK(cudaGetLastError());

            update_centroids<<<blocks_clusters, threads>>>(
                d_centroids, d_centroid_sums, d_cluster_counts, k, num_cols);
            CUDA_CHECK(cudaGetLastError());
        } else {
            converged = 1;
            printf("Convergiu apos %d iteracoes\n", iterations);
        }

        iterations++;
    }

    // transferencia final GPU -> CPU dos rotulos para exportacao
    CUDA_CHECK(cudaEventRecord(ev_d2h_start));
    CUDA_CHECK(cudaMemcpy(assignments, d_assignments,
                          row_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));

    clock_gettime(CLOCK_MONOTONIC, &tempo_fim);
    double total_time = (tempo_fim.tv_sec - tempo_inicio.tv_sec)
                      + (tempo_fim.tv_nsec - tempo_inicio.tv_nsec) / 1e9;

    float h2d_ms = 0.0f, d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_stop));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n--- metricas do cuda kmeans ---\n");
    printf("GPU: %s\n", prop.name);
    printf("Arquivo de entrada: %s\n", original_filename);
    printf("Numero de clusters (k): %d\n", k);
    printf("Numero de iteracoes: %d\n", iterations);
    printf("Tempo de transferencia CPU->GPU: %f ms\n", h2d_ms);
    printf("Tempo de transferencia GPU->CPU: %f ms\n", d2h_ms);
    printf("Tempo total de execucao: %f segundos\n", total_time);

    export_dataset_with_clusters(
        original_filename,
        "clusters_cuda_kmeans.csv",
        assignments,
        row_count,
        start_row
    );

    cudaEventDestroy(ev_h2d_start);
    cudaEventDestroy(ev_h2d_stop);
    cudaEventDestroy(ev_d2h_start);
    cudaEventDestroy(ev_d2h_stop);

    cudaFree(d_dataset);
    cudaFree(d_centroids);
    cudaFree(d_centroid_sums);
    cudaFree(d_assignments);
    cudaFree(d_cluster_counts);
    cudaFree(d_changed);

    free(flat_centroids);
    free(assignments);
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

    cuda_kmeans(k, dataset, row_count, filename, start_row, start_col, end_col);

    free(dataset);
    return 0;
}
