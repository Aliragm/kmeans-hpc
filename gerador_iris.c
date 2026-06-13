#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// gerar arquivo csv simulando o dataset iris
void gerar_csv(const char* nome_arquivo, int qtd_amostras) {
    FILE *file = fopen(nome_arquivo, "w");
    if (file == NULL) {
        printf("erro: falha ao criar arquivo %s\n", nome_arquivo);
        return;
    }

    // manter cabecalho padrao do dataset original para compatibilidade
    fprintf(file, "SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species\n");

    srand(time(NULL));
    const char* especies[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    for (int i = 0; i < qtd_amostras; i++) {
        // gerar valores aleatorios baseados nos limites biologicos das especies
        float sepal_l = 4.0 + ((float)rand() / RAND_MAX) * 4.0;
        float sepal_w = 2.0 + ((float)rand() / RAND_MAX) * 2.5;
        float petal_l = 1.0 + ((float)rand() / RAND_MAX) * 6.0;
        float petal_w = 0.1 + ((float)rand() / RAND_MAX) * 2.5;
        
        // associar amostra aleatoriamente a uma das tres classes existentes
        int especie_id = rand() % 3;

        fprintf(file, "%.1f,%.1f,%.1f,%.1f,%s\n", sepal_l, sepal_w, petal_l, petal_w, especies[especie_id]);
    }

    fclose(file);
    printf("-> %s gerado: %d amostras\n", nome_arquivo, qtd_amostras);
}

int main() {
    printf("--- inicio: data augmentation (iris) ---\n");
    gerar_csv("Iris_10k.csv", 10000);
    gerar_csv("Iris_100k.csv", 100000);
    gerar_csv("Iris_1M.csv", 1000000);
    printf("--- fim: geracao de arquivos ---\n");
    return 0;
}
