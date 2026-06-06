#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_SIZE 1024

char*** read_csv(const char* FILENAME, int* out_row_count) {
    char*** dataset = NULL;
    int counter = 0;

    FILE *file = fopen(FILENAME, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    char line[MAX_LINE_SIZE];
    
    //descomentar para tirar o cabecalho do arquivo
    /*if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return NULL; 
    }*/

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