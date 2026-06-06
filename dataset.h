#ifndef DATASET_H
#define DATASET_H

char*** read_csv(const char* FILENAME, int* out_row_count);
void free_dataset(char*** dataset, int row_count);

#endif