#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"

void sequential_kmeans(int k, char*** dataset, int row_count){
    int dimensions = 0;

    while (dataset[1][dimensions] != NULL) {
        printf("[%s] ", dataset[1][dimensions]);
        dimensions++;
    }
    printf("\n");

    int* centroids = malloc(k * ((dimensions-1) * sizeof(char)));
    int converged = 0;//false

    while(converged == 0){
        //continuar a implementação...
    }


}

int main(int argc, char *argv[]){
    int row_count = 0;
    char*** dataset = read_csv("Iris.csv", &row_count);

    if (dataset == NULL) {
        printf("Error at loading the dataset.\n");
        return 1;
    }

    sequential_kmeans(3, dataset, row_count);

    free_dataset(dataset, row_count);

    return 0;
}