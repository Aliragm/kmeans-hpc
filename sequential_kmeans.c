#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "dataset.h"

void sequential_kmeans(int k, char*** dataset, int row_count, const char* original_filename, int start_row, int start_col, int end_col){
    if (row_count <= 0 || dataset == NULL) return;

    int* assignments = malloc(row_count * sizeof(int));
    for (int i = 0; i < row_count; i++) {
        assignments[i] = -1;
    }

    float** centroids = malloc(k * sizeof(float*));
    generate_centroids(k, dataset, centroids, row_count, start_col, end_col);

    int converged = 0;
    int iterations = 0;

    while (converged == 0) {
        int changed = 0;
        
        for (int i = 0; i < row_count; i++) {
            char** point = dataset[i];
            
            int new_cluster = assimilate_to_centroid(point, centroids, k, start_col, end_col);
            
            if (new_cluster != assignments[i]) {
                assignments[i] = new_cluster;
                changed = 1;
            }
        } 

        if (changed == 1) {
            update_centroids(dataset, centroids, assignments, k, row_count, start_col, end_col);
        } else {
            converged = 1;
            printf("The algorithm converged after %d iterations\n", iterations);
        }

        iterations++;
        if (iterations > 300) { 
            printf("Alert: limit of iterations reached (300).\n");
            break; 
        }
    }

    export_dataset_with_clusters(original_filename, "clusters_sequential_kmeans.csv", assignments, row_count, start_row);

    free_centroids(k, centroids);
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