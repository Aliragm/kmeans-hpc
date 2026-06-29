# compilador e flags
mpicc   = mpicc
cc      = gcc
nvc     = nvc
nvcc    = nvcc
cflags  = -O3 -Wall -Wextra
libs    = -lm

# arquivos fonte
src     = sequential_kmeans.c dataset.c
bin     = kmeans_otimizado

# alvos do gerador de dados
gerador_src = gerador_iris.c
gerador_bin = gerador

mpi_openmp_src = mpi_openmp_kmeans.c dataset.c
mpi_openmp_bin = mpi_openmp_kmeans

openmp_gpu_src = openmp_gpu_kmeans.c dataset.c
openmp_gpu_bin = openmp_gpu_kmeans

cuda_src = cuda_kmeans.cu
cuda_bin = cuda_kmeans

# padrao: compila o kmeans otimizado
all: $(bin)

# linkagem final
$(bin): $(src:.c=.o)
	$(cc) $(cflags) $^ -o $@ $(libs)

# compilacao dos .c para .o
%.o: %.c dataset.h
	$(cc) $(cflags) -c $< -o $@

# compila o gerador de datasets sinteticos
gerador: $(gerador_src)
	$(cc) $(cflags) $^ -o $(gerador_bin) $(libs)

# gera os datasets sinteticos (10k, 100k, 1m)
datasets: gerador
	./$(gerador_bin)

mpi_openmp: $(mpi_openmp_src)
	$(mpicc) $(cflags) $^ -o $(mpi_openmp_bin) -fopenmp

openmp_gpu: $(openmp_gpu_src)
	$(nvc) -mp=gpu -Minfo=mp -O2 $^ -o $(openmp_gpu_bin)

# compila a versao CUDA. dataset.o e gerado pelo gcc (linkagem C) e linkado
# com o objeto do kernel produzido pelo nvcc
cuda: $(cuda_src) dataset.o
	$(nvcc) -O3 $^ -o $(cuda_bin) $(libs)

# executa o kmeans com o iris original
run: $(bin)
	./$(bin)

# limpa arquivos compilados e saidas
clean:
	rm -f *.o $(bin) $(gerador_bin) $(mpi_openmp_bin) $(openmp_gpu_bin) $(cuda_bin) clusters_*.csv

.PHONY: all gerador datasets run clean mpi_openmp openmp_gpu cuda
