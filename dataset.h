#ifndef DATASET_H
#define DATASET_H

float* read_csv_to_floats(const char* FILENAME, int* out_row_count, int start_col, int end_col, int without_header);
void generate_centroids(int k, float* flat_dataset, float* flat_centroids, int row_count, int num_cols);
void export_dataset_with_clusters(const char* original_filename, const char* output_filename, int* assignments, int row_count, int start_row);

#endif