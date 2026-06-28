# Relatório Comparativo de Desempenho: K-Means

Este relatório apresenta os resultados obtidos ao executar o algoritmo K-Means em três estratégias (Sequencial, MPI + OpenMP híbrido e OpenMP com offloading para GPU) para diferentes tamanhos de bases de dados.

## Tabela Comparativa de Tempos

O tempo de execução é dado em segundos. O **Speedup** total é baseado em $T_{seq} / T_{paralelo}$.

| Dataset | Qtd. Amostras | Sequencial (s) | MPI + OpenMP (s) | Speedup MPI | OpenMP GPU (s) | Speedup GPU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Iris** | 150 | 0,000285 | 0,157297 | 0,0018x | 0,340270 | 0,0008x |
| **Iris_10k** | 10.000 | 0,001394 | 0,296530 | 0,0047x | 0,330942 | 0,0042x |
| **Iris_100k** | 100.000 | 0,008714 | 0,226757 | 0,0384x | 0,337203 | 0,0258x |
| **Iris_1M** | 1.000.000 | 0,080180 | 0,226772 | 0,3536x | 0,388385 | 0,2064x |
| **Iris_10M** | 10.000.000 | 2,493860 | 1,554380 | 1,6044x | 0,727277 | 3,4290x |
| **Iris_50M** | 50.000.000 | 13,043533 | 7,540540 | 1,7297x | 3,689345 | 3,5354x |
| **Iris_100M**\* | 100.000.000 | 509,634190 | 11,782815 | 43,2523x\* | 188,345879 | 2,7058x\* |

> **Nota sobre o Dataset de 100M:** A inicialização aleatória dos centróides fez com que as execuções **Sequencial** e **GPU** não convergissem, atingindo o limite de **300 iterações**, enquanto a execução **MPI + OpenMP** convergiu em apenas **9 iterações**. Para uma comparação justa, veja a análise do tempo por iteração abaixo.

---

## Tempo por Iteração (Dataset de 100M)

Para anular o efeito da aleatoriedade do número de iterações até a convergência, comparamos o tempo gasto por iteração no dataset de 100M:

* **Sequencial:** $509,634\text{ s} / 300\text{ iterações} = \mathbf{1,698\text{ s / iteração}}$
* **MPI + OpenMP (20 threads):** $11,782\text{ s} / 9\text{ iterações} = \mathbf{1,309\text{ s / iteração}}$ (Speedup real de **1,29x**)
* **OpenMP GPU:** $188,345\text{ s} / 300\text{ iterações} = \mathbf{0,627\text{ s / iteração}}$ (Speedup real de **2,70x**)

---

## Conclusões do Estudo de Caso

1. **Ponto de Cruzamento (Crossover Point):**
   * Até **1M de amostras**, a versão sequencial é mais rápida porque o custo de computação pura é inferior ao custo de inicialização das threads, processos MPI e cópia de memória para a GPU.
   * A partir de **10M de amostras**, a paralelização começa a compensar. A GPU assume a liderança clara com um speedup de **~3.4x** em relação ao sequencial.

2. **Gargalo de Escrita Assíncrona e I/O:**
   * A exportação do arquivo CSV final de clusters (`clusters_*.csv`) é sequencial. Para 100M de linhas, gerar esse arquivo de saída leva muito tempo e consome espaço significativo de I/O em disco, o que infla o tempo total.

3. **Gargalo Atômico na GPU:**
   * Embora a GPU execute a **fase 1** (atribuição de clusters) de forma massivamente paralela em menos de 1 milissegundo, a **fase 2** (soma dos centróides) é limitada por `#pragma omp atomic` em apenas $k=3$ variáveis. Conforme a quantidade de threads cresce de 10M para 100M, a contenção nessas operações atômicas impede que a GPU obtenha um speedup linear maior (ficando em 2,70x por iteração).
