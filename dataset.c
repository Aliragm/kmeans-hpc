#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

#define MAX_LINE_SIZE 1024

float* read_csv_to_floats(const char* FILENAME, int* out_row_count, int start_col, int end_col, int without_header) {
    FILE* file = fopen(FILENAME, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    int num_cols = end_col - start_col + 1;
    int capacity = 1024;
    float* data = malloc(capacity * num_cols * sizeof(float));
    int row_count = 0;
    char line[MAX_LINE_SIZE];

    if (without_header) {
        if (fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            free(data);
            return NULL;
        }
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        line[strcspn(line, "\r\n")] = '\0';

        if (row_count >= capacity) {
            capacity *= 2;
            data = realloc(data, capacity * num_cols * sizeof(float));
        }

        int col = 0;
        int data_idx = 0;
        char* token = strtok(line, ",");
        while (token != NULL) {
            if (col >= start_col && col <= end_col) {
                data[row_count * num_cols + data_idx] = atof(token);
                data_idx++;
            }
            col++;
            token = strtok(NULL, ",");
        }

        if (data_idx != num_cols) {
            fprintf(stderr, "Warning: row %d has %d columns, expected %d\n", row_count + 1, data_idx, num_cols);
        }

        row_count++;
    }

    fclose(file);

    data = realloc(data, row_count * num_cols * sizeof(float));
    *out_row_count = row_count;
    return data;
}

void generate_centroids(int k, float* flat_dataset, float* flat_centroids, int row_count, int num_cols) {
    float* min_values = malloc(num_cols * sizeof(float));
    float* max_values = malloc(num_cols * sizeof(float));

    for (int j = 0; j < num_cols; j++) {
        min_values[j] = FLT_MAX;
        max_values[j] = -FLT_MAX;
    }

    for (int i = 0; i < row_count; i++) {
        for (int j = 0; j < num_cols; j++) {
            float val = flat_dataset[i * num_cols + j];
            if (val < min_values[j]) min_values[j] = val;
            if (val > max_values[j]) max_values[j] = val;
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < num_cols; j++) {
            float min = min_values[j];
            float max = max_values[j];
            float random_factor = (float)rand() / (float)RAND_MAX;
            flat_centroids[i * num_cols + j] = min + random_factor * (max - min);
        }
    }

    free(min_values);
    free(max_values);
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
        } else if (assignment_idx < row_count) {
            fprintf(outfile, "%s,%d\n", line, assignments[assignment_idx]);
            assignment_idx++;
        } else {
            fprintf(outfile, "%s,\n", line);
        }
        current_file_row++;
    }

    fclose(infile);
    fclose(outfile);
    printf("File '%s' generated successfully.\n", output_filename);
}
