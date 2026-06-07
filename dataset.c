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

void generate_centroids(int k, char*** dataset, float** centroids, int row_count, int start_col, int end_col) {
    int total_cols = end_col - start_col + 1;
    float* min_values = malloc(total_cols * sizeof(float));
    float* max_values = malloc(total_cols * sizeof(float));

    for (int j = 0; j < total_cols; j++) {
        min_values[j] = FLT_MAX;
        max_values[j] = -FLT_MAX;
    }

    for (int i = 0; i < row_count; i++) {
        for (int j = start_col; j <= end_col; j++) {
            if (dataset[i][j] != NULL) {
                float current_number = strtof(dataset[i][j], NULL);
                int local_col = j - start_col;
                
                if (current_number < min_values[local_col]) min_values[local_col] = current_number;
                if (current_number > max_values[local_col]) max_values[local_col] = current_number;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        centroids[i] = malloc(total_cols * sizeof(float)); 

        for (int j = 0; j < total_cols; j++) {
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

int assimilate_to_centroid(char** point, float** centroids, int k, int start_col, int end_col) {
    int closest_centroid = -1;
    float min_distance = FLT_MAX;

    for (int c = 0; c < k; c++) {
        float current_distance = 0.0;

        // Loop genérico usando as colunas demarcadas
        for (int d = start_col; d <= end_col; d++) {
            if (point[d] != NULL) {
                float point_value = atof(point[d]);
                // Mapeia a coluna do dataset para o índice local do centroide
                float centroid_value = centroids[c][d - start_col]; 
                
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

void update_centroids(char*** dataset, float** centroids, int* assignments, int k, int row_count, int start_col, int end_col) {
    float** new_sums = malloc(k * sizeof(float*));
    int* cluster_sizes = calloc(k, sizeof(int));

    for (int i = 0; i < k; i++) {
        // Aloca espaço baseado no intervalo de colunas que estamos usando
        new_sums[i] = calloc((end_col - start_col + 1), sizeof(float));
    }

    for (int i = 0; i < row_count; i++) {
        int cluster_id = assignments[i];
        
        if (cluster_id >= 0 && cluster_id < k) {
            cluster_sizes[cluster_id]++;
            
            // Loop corre apenas no intervalo das colunas desejadas
            for (int j = start_col; j <= end_col; j++) {
                if (dataset[i][j] != NULL) {
                    // Ajustamos o índice para gravar a partir do 0 na matriz local do centroide
                    new_sums[cluster_id][j - start_col] += atof(dataset[i][j]);
                }
            }
        }
    }

    for (int c = 0; c < k; c++) {
        if (cluster_sizes[c] > 0) {
            for (int j = start_col; j <= end_col; j++) {
                centroids[c][j - start_col] = new_sums[c][j - start_col] / cluster_sizes[c];
            }
        } 
    }

    for (int i = 0; i < k; i++) {
        free(new_sums[i]);
    }
    free(new_sums);
    free(cluster_sizes);
}

void export_dataset_with_clusters(const char* original_filename, const char* output_filename, int* assignments, int row_count, int start_row) {
    FILE* infile = fopen(original_filename, "r");
    FILE* outfile = fopen(output_filename, "w");

    if (infile == NULL || outfile == NULL) {
        fprintf(stderr, "Error opening files for exporting.\n");
        if (infile) fclose(infile);
        if (outfile) fclose(outfile);
        return;
    }

    char line[MAX_LINE_SIZE];
    int current_file_row = 0;
    int assignment_idx = 0;

    if (fgets(line, sizeof(line), infile) != NULL) {
        line[strcspn(line, "\r\n")] = '\0';
        fprintf(outfile, "%s,cluster\n", line);
        current_file_row++;
    }

    while (fgets(line, sizeof(line), infile) != NULL) {
        line[strcspn(line, "\r\n")] = '\0';

        if (current_file_row < start_row) {
            fprintf(outfile, "%s,\n", line);
        } 

        else if (assignment_idx < row_count) {
            fprintf(outfile, "%s,%d\n", line, assignments[assignment_idx]);
            assignment_idx++;
        } 
    
        else {
            fprintf(outfile, "%s,\n", line);
        }
        current_file_row++;
    }

    fclose(infile);
    fclose(outfile);
    printf("File '%s' generated successfully.\n", output_filename);
}