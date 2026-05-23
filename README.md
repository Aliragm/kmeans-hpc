# Paralelização do Algoritmo K-Means

Este repositório contém o projeto da disciplina de Computação de Alto Desempenho, com foco na implementação, paralelização e análise de desempenho do algoritmo de clusterização **K-Means** escrito em C.

O projeto utiliza o clássico **Iris dataset** como base de dados padronizada para avaliar tanto o ganho de desempenho computacional quanto a qualidade matemática dos clusters gerados em diferentes arquiteturas de hardware.

## Equipe
* André Lira Gonçalves de Medeiros
* Arthur Emanuel Souza Cassiano da Costa
* Jose Cassio de Araujo Goes
* Luis Felipe Medeiros Reis
* Mateus Sousa da Silva

## Objetivos e Fases de Implementação
O projeto está dividido em quatro implementações principais para fins de comparação direta:

1. **Sequencial (Baseline):** Implementação base em C para validação lógica e coleta do tempo de execução de referência.
2. **Híbrida (MPI + OpenMP):** Paralelização para sistemas de memória distribuída e compartilhada.
3. **OpenMP em GPU:** Uso da GPU utilizando diretivas OpenMP.
4. **CUDA:** Implementação nativa em GPU buscando o máximo aproveitamento da arquitetura massivamente paralela.

## Métricas Analisadas
Para garantir uma avaliação completa, o projeto extrai dois grupos de métricas:

* **Métricas de Desempenho Computacional:**
  * Tempo de Execução
  * Speedup
  * Eficiência
  * Escalabilidade Forte e Fraca
* **Métricas de Qualidade (Diferencial):**
  * Validação da qualidade dos clusters gerados no Iris dataset (ex: variância interna, separação), assegurando que as otimizações paralelas não comprometam o resultado matemático do algoritmo.
