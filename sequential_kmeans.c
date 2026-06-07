#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "dataset.h"

void sequential_kmeans(int k, char*** dataset, int row_count){

    if (row_count <= 0 || dataset == NULL) return;

    int dimensions = 0;

    int* assignments = malloc(row_count * sizeof(int));
    for (int i = 0; i < row_count; i++) {
        assignments[i] = -1;
    }

    while (dataset[0][dimensions] != NULL) {
        dimensions++;
    }

    float** centroids = malloc(k * sizeof(float*));
    generate_centroids(k, dataset, centroids, dimensions, row_count);

    int converged = 0;//false
    int iterations = 0;

    while (converged == 0) {
        int changed = 0;
        
        for (int i = 0; i < row_count; i++) {
            char** point = dataset[i];
            
            int new_cluster = assimilate_to_centroid(point, centroids, k, dimensions);
            
            if (new_cluster != assignments[i]) {
                assignments[i] = new_cluster;
                changed = 1;
            }
        } 

        if (changed == 1) {
            update_centroids(dataset, centroids, assignments, k, dimensions, row_count);
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

    free_centroids(k, centroids);
    free(assignments);
}

int main(int argc, char *argv[]){

    srand(time(NULL));

    int row_count = 0;
    char*** dataset = read_csv("Iris.csv", &row_count, 1);

    if (dataset == NULL) {
        printf("Error at loading the dataset.\n");
        return 1;
    }

    sequential_kmeans(3, dataset, row_count);

    free_dataset(dataset, row_count);

    return 0;
}