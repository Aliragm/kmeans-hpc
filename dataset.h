#ifndef DATASET_H
#define DATASET_H

char*** read_csv(const char* FILENAME, int* out_row_count, int without_header);
void free_dataset(char*** dataset, int row_count);
void generate_centroids(int k, char*** dataset, float** centroids, int row_count, int start_col, int end_col);
void free_centroids(int k, float** centroids);
int assimilate_to_centroid(char** point, float** centroids, int k, int start_col, int end_col);
void update_centroids(char*** dataset, float** centroids, int* assignments, int k, int row_count, int start_col, int end_col);
void export_dataset_with_clusters(const char* original_filename, const char* output_filename, int* assignments, int row_count, int start_row);

#endif