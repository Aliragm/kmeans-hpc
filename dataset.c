#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

#define MAX_LINE_SIZE 1024

char*** read_csv(const char* FILENAME, int* out_row_count, int without_header) {

    if(!(without_header == 0 || without_header == 1)){
        fprintf(stderr, "Error: without_header must be 0 (false) or 1 (true).\n");
        return NULL;        
    }

    char*** dataset = NULL;
    int counter = 0;

    FILE *file = fopen(FILENAME, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    char line[MAX_LINE_SIZE];
    
    //descomentar para tirar o cabecalho do arquivo
    if(without_header == 1){
        if (fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return NULL; 
        }
    }
    

    while (fgets(line, sizeof(line), file) != NULL) {
        int internal_counter = 0;
        char** dataset_line = NULL;

        line[strcspn(line, "\r\n")] = '\0';

        char *token = strtok(line, ",");
        while (token != NULL) {
            dataset_line = realloc(dataset_line, (internal_counter + 1) * sizeof(char*));
            
            dataset_line[internal_counter] = strdup(token); 
            
            internal_counter++;
            token = strtok(NULL, ",");
        }

        dataset_line = realloc(dataset_line, (internal_counter + 1) * sizeof(char*));
        dataset_line[internal_counter] = NULL;

        dataset = realloc(dataset, (counter + 1) * sizeof(char**));
        dataset[counter] = dataset_line;

        counter++;
    }

    fclose(file);
    
    *out_row_count = counter; 
    return dataset;
}

void free_dataset(char*** dataset, int row_count){
    for (int i = 0; i < row_count; i++) {
        int j = 0;
        while (dataset[i][j] != NULL) {
            free(dataset[i][j]);
            j++;
        }
        free(dataset[i]);
    }
    free(dataset);
}

void generate_centroids(int k, char*** dataset, float** centroids, int dimensions, int row_count) {
    
    float* min_values = malloc(dimensions * sizeof(float));
    float* max_values = malloc(dimensions * sizeof(float));

    for (int j = 0; j < dimensions; j++) {
        min_values[j] = FLT_MAX;
        max_values[j] = -FLT_MAX;
    }

    for (int i = 0; i < row_count; i++) {
        for (int j = 0; j < dimensions; j++) {
            if (dataset[i][j] != NULL) {
                float current_number = strtof(dataset[i][j], NULL);
                
                if (current_number < min_values[j]) min_values[j] = current_number;
                if (current_number > max_values[j]) max_values[j] = current_number;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        centroids[i] = malloc(dimensions * sizeof(float)); 

        for (int j = 0; j < dimensions; j++) {
            float min = min_values[j];
            float max = max_values[j];

            float random_factor = (float)rand() / (float)RAND_MAX;
            centroids[i][j] = min + random_factor * (max - min);
        }
    }

    free(min_values);
    free(max_values);
}

void free_centroids(int k, float** centroids){
    for(int i = 0; i < k; i++) {
        free(centroids[i]);
    }
    free(centroids);
}

int assimilate_to_centroid(char** point, float** centroids, int k, int dimensions) {
    int closest_centroid = -1;
    float min_distance = FLT_MAX;

    for (int c = 0; c < k; c++) {
        float current_distance = 0.0;

        for (int d = 0; d < dimensions; d++) {
            if (point[d] != NULL) {
                float point_value = atof(point[d]);
                float centroid_value = centroids[c][d];
                
                float diff = point_value - centroid_value;
                current_distance += diff * diff;
            }
        }

        if (current_distance < min_distance) {
            min_distance = current_distance;
            closest_centroid = c;
        }
    }

    return closest_centroid;
}

void update_centroids(char*** dataset, float** centroids, int* assignments, int k, int dimensions, int row_count) {
    float** new_sums = malloc(k * sizeof(float*));
    int* cluster_sizes = calloc(k, sizeof(int));

    for (int i = 0; i < k; i++) {
        new_sums[i] = calloc(dimensions, sizeof(float));
    }

    for (int i = 0; i < row_count; i++) {
        int cluster_id = assignments[i];
        
        if (cluster_id >= 0 && cluster_id < k) {
            cluster_sizes[cluster_id]++;
            
            for (int j = 0; j < dimensions; j++) {
                if (dataset[i][j] != NULL) {
                    new_sums[cluster_id][j] += atof(dataset[i][j]);
                }
            }
        }
    }

    for (int c = 0; c < k; c++) {
        if (cluster_sizes[c] > 0) {
            for (int j = 0; j < dimensions; j++) {
                centroids[c][j] = new_sums[c][j] / cluster_sizes[c];
            }
        } 
    }

    for (int i = 0; i < k; i++) {
        free(new_sums[i]);
    }
    free(new_sums);
    free(cluster_sizes);
}