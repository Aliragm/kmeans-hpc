#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import re
import time

# Configurações de cores para o terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

def log_info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")

def log_success(msg):
    print(f"{GREEN}[SUCCESS]{RESET} {msg}")

def log_warning(msg):
    print(f"{YELLOW}[WARNING]{RESET} {msg}")

def log_error(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")

# Definição dos datasets e quantidade de amostras
DATASETS = [
    {"file": "Iris.csv", "samples": 150, "display": "Iris (150)"},
    {"file": "Iris_10k.csv", "samples": 10000, "display": "Iris 10K"},
    {"file": "Iris_100k.csv", "samples": 100000, "display": "Iris 100K"},
    {"file": "Iris_1M.csv", "samples": 1000000, "display": "Iris 1M"},
    {"file": "Iris_10M.csv", "samples": 10000000, "display": "Iris 10M"},
    {"file": "Iris_50M.csv", "samples": 50000000, "display": "Iris 50M"},
    {"file": "Iris_100M.csv", "samples": 100000000, "display": "Iris 100M"}
]

# Definição dos scripts Slurm para cada implementação
IMPLEMENTATIONS = [
    {"name": "Sequencial", "script": "scripts/sequential_kmeans.sh"},
    {"name": "MPI + OpenMP", "script": "scripts/mpi_openmp_kmeans.sh"},
    {"name": "OpenMP GPU", "script": "scripts/openmp_gpu_kmeans.sh"},
    {"name": "CUDA", "script": "scripts/cuda_kmeans.sh"}
]

def run_command_local(cmd, shell=False):
    """Executa um comando e retorna o código de retorno, stdout e stderr."""
    try:
        res = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        return 0, res.stdout, res.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr

def check_and_compile():
    log_info("Iniciando a compilação do projeto...")
    # Executa o make para compilar o sequencial, MPI e o gerador de dados
    # Nota: As versões OpenMP GPU e CUDA serão compiladas dinamicamente nos compute nodes via seus respectivos scripts .sh
    code, stdout, stderr = run_command_local(["make", "clean"])
    code, stdout, stderr = run_command_local(["make", "gerador", "all", "mpi_openmp"])
    
    if code != 0:
        log_error("Falha ao compilar os arquivos base!")
        print(stderr)
        sys.exit(1)
    log_success("Compilação concluída com sucesso (Gerador, Sequencial e MPI).")

def check_and_generate_datasets():
    log_info("Verificando se os datasets existem...")
    missing_any = False
    for ds in DATASETS:
        if not os.path.exists(ds["file"]):
            log_warning(f"Dataset {ds['file']} não encontrado.")
            missing_any = True
            
    if missing_any:
        log_info("Gerando datasets sintéticos rodando o gerador...")
        # Compila se necessário
        if not os.path.exists("./gerador"):
            run_command_local(["make", "gerador"])
        
        # Executa o gerador para criar os datasets
        code, stdout, stderr = run_command_local(["./gerador"])
        if code != 0:
            log_error("Falha ao gerar os datasets!")
            print(stderr)
            sys.exit(1)
        log_success("Todos os datasets sintéticos foram gerados com sucesso.")
    else:
        log_success("Todos os datasets já estão presentes.")

def parse_output_file(filepath):
    """Lê o arquivo de saída do job e extrai o tempo de execução e iterações."""
    if not os.path.exists(filepath):
        return None, None, {}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Regex para tempo de execução (segundos)
    # Suporta: "tempo de execucao: X segundos", "Tempo total de execução: X segundos", "Tempo total de execucao: X segundos"
    time_match = re.search(r'(?:tempo de execucao|Tempo total de execu[çc][ãa]o):\s*([\d\.]+)', content, re.IGNORECASE)
    
    # Regex para número de iterações
    # Suporta: "Numero de iteracoes: X", "Número de iterações: X", ou "Convergiu apos X iteracoes"
    iter_match = re.search(r'(?:Numero de itera[çc][õo]es|N[úu]mero de itera[çc][õo]es):\s*(\d+)', content, re.IGNORECASE)
    if not iter_match:
        iter_match = re.search(r'Convergiu apos\s*(\d+)\s*itera[çc][õo]es', content, re.IGNORECASE)

    exec_time = float(time_match.group(1)) if time_match else None
    iterations = int(iter_match.group(1)) if iter_match else None

    # Métricas extras específicas para CUDA (tempo de transferência)
    extras = {}
    h2d_match = re.search(r'Tempo de transferencia CPU->GPU:\s*([\d\.]+)\s*ms', content, re.IGNORECASE)
    d2h_match = re.search(r'Tempo de transferencia GPU->CPU:\s*([\d\.]+)\s*ms', content, re.IGNORECASE)
    
    if h2d_match:
        extras['h2d_ms'] = float(h2d_match.group(1))
    if d2h_match:
        extras['d2h_ms'] = float(d2h_match.group(1))

    return exec_time, iterations, extras

def main():
    check_and_compile()
    check_and_generate_datasets()

    results = [] # Lista de dicionários para armazenar todos os resultados estruturados
    
    # Arquivo temporário padrão usado para os outputs do sbatch
    # %j será substituído pelo ID do job pelo Slurm
    temp_output_pattern = "temp_exp_job_%j.out"

    total_runs = len(DATASETS) * len(IMPLEMENTATIONS)
    current_run = 0

    log_info(f"Iniciando a execução sequencial de {total_runs} experimentos...")

    for ds in DATASETS:
        ds_file = ds["file"]
        ds_samples = ds["samples"]
        ds_display = ds["display"]

        log_info(f"--- Processando tamanho de dataset: {ds_display} ({ds_samples} amostras) ---")

        for impl in IMPLEMENTATIONS:
            impl_name = impl["name"]
            script_path = impl["script"]
            current_run += 1

            log_info(f"[{current_run}/{total_runs}] Submetendo {impl_name} para o dataset {ds_file}...")

            # Comando para submeter o Slurm e aguardar a execução (--wait)
            # Sobrescrevemos o arquivo de saída para rastrear facilmente
            cmd = ["sbatch", "--wait", "-o", temp_output_pattern, script_path, ds_file]
            
            start_submit_time = time.time()
            code, stdout, stderr = run_command_local(cmd)
            elapsed_submit = time.time() - start_submit_time

            if code != 0:
                log_error(f"Erro ao submeter o job {impl_name} para o dataset {ds_file}!")
                print(stderr)
                results.append({
                    "dataset": ds_display,
                    "samples": ds_samples,
                    "impl": impl_name,
                    "time": None,
                    "iterations": None,
                    "status": "FALHA_SUBMISSAO",
                    "extras": {}
                })
                continue

            # O output do sbatch contem o Job ID: "Submitted batch job 12345"
            job_id_match = re.search(r'Submitted batch job (\d+)', stdout)
            if not job_id_match:
                log_error(f"Não foi possível obter o Job ID da saída: '{stdout.strip()}'")
                results.append({
                    "dataset": ds_display,
                    "samples": ds_samples,
                    "impl": impl_name,
                    "time": None,
                    "iterations": None,
                    "status": "ERRO_JOBID",
                    "extras": {}
                })
                continue

            job_id = job_id_match.group(1)
            job_output_file = f"temp_exp_job_{job_id}.out"

            # Parsear o resultado obtido no arquivo de saída do Slurm
            exec_time, iterations, extras = parse_output_file(job_output_file)

            if exec_time is not None:
                log_success(f"-> Concluído. Tempo: {exec_time:.6f}s | Iterações: {iterations}")
                results.append({
                    "dataset": ds_display,
                    "samples": ds_samples,
                    "impl": impl_name,
                    "time": exec_time,
                    "iterations": iterations,
                    "status": "OK",
                    "extras": extras
                })
            else:
                log_error(f"-> Falha ao ler métricas do arquivo {job_output_file} (pode ter ocorrido erro no job)")
                results.append({
                    "dataset": ds_display,
                    "samples": ds_samples,
                    "impl": impl_name,
                    "time": None,
                    "iterations": None,
                    "status": "ERRO_METRICAS",
                    "extras": {}
                })

            # Deletar arquivo temporário para não poluir o diretório
            if os.path.exists(job_output_file):
                try:
                    os.remove(job_output_file)
                except Exception as e:
                    log_warning(f"Não foi possível remover {job_output_file}: {e}")

    # --- Consolidação dos Resultados ---
    log_info("Todos os jobs foram executados. Consolidando resultados...")

    # Gerar arquivo CSV
    csv_file = "resultados_experimentos.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Dataset,Qtd_Amostras,Implementacao,Tempo_s,Iteracoes,Status,H2D_Transfer_ms,D2H_Transfer_ms\n")
        for r in results:
            t = f"{r['time']:.6f}" if r["time"] is not None else "N/A"
            it = str(r["iterations"]) if r["iterations"] is not None else "N/A"
            h2d = f"{r['extras'].get('h2d_ms', 'N/A')}"
            d2h = f"{r['extras'].get('d2h_ms', 'N/A')}"
            f.write(f"{r['dataset']},{r['samples']},{r['impl']},{t},{it},{r['status']},{h2d},{d2h}\n")
    log_success(f"Arquivo CSV gerado: {csv_file}")

    # Gerar arquivo Markdown (Tabela Comparativa com Speedups)
    md_file = "resultados_experimentos.md"
    
    # Agrupar resultados por tamanho de dataset para o cálculo de speedup
    grouped_results = {}
    for r in results:
        ds = r["dataset"]
        if ds not in grouped_results:
            grouped_results[ds] = {}
        grouped_results[ds][r["impl"]] = r

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Relatório Consolidado de Desempenho: K-Means\n\n")
        f.write("Este relatório foi gerado automaticamente após a execução sequencial de todos os experimentos em lote na infraestrutura de HPC.\n\n")
        
        # Tabela Principal
        f.write("## Tabela Comparativa de Tempos\n\n")
        f.write("| Dataset | Qtd. Amostras | Sequencial (s) | MPI + OpenMP (s) | Speedup MPI | OpenMP GPU (s) | Speedup GPU | CUDA (s) | Speedup CUDA |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for ds_display in [d["display"] for d in DATASETS]:
            if ds_display not in grouped_results:
                continue
            
            ds_group = grouped_results[ds_display]
            samples = ds_group.get("Sequencial", {}).get("samples", 0)
            
            # Obter tempos
            t_seq = ds_group.get("Sequencial", {}).get("time")
            t_mpi = ds_group.get("MPI + OpenMP", {}).get("time")
            t_omp_gpu = ds_group.get("OpenMP GPU", {}).get("time")
            t_cuda = ds_group.get("CUDA", {}).get("time")
            
            # Formatar tempos
            t_seq_str = f"{t_seq:.6f}" if t_seq is not None else "N/A"
            t_mpi_str = f"{t_mpi:.6f}" if t_mpi is not None else "N/A"
            t_omp_gpu_str = f"{t_omp_gpu:.6f}" if t_omp_gpu is not None else "N/A"
            t_cuda_str = f"{t_cuda:.6f}" if t_cuda is not None else "N/A"
            
            # Calcular Speedups
            speedup_mpi = f"{t_seq / t_mpi:.2f}x" if (t_seq and t_mpi) else "N/A"
            speedup_gpu = f"{t_seq / t_omp_gpu:.2f}x" if (t_seq and t_omp_gpu) else "N/A"
            speedup_cuda = f"{t_seq / t_cuda:.2f}x" if (t_seq and t_cuda) else "N/A"
            
            f.write(f"| **{ds_display}** | {samples:,} | {t_seq_str} | {t_mpi_str} | {speedup_mpi} | {t_omp_gpu_str} | {speedup_gpu} | {t_cuda_str} | {speedup_cuda} |\n")
            
        f.write("\n---\n\n")
        
        # Tabela de Iterações
        f.write("## Tabela de Iterações até Convergência\n\n")
        f.write("| Dataset | Qtd. Amostras | Iterações Seq | Iterações MPI | Iterações OpenMP GPU | Iterações CUDA |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for ds_display in [d["display"] for d in DATASETS]:
            if ds_display not in grouped_results:
                continue
            ds_group = grouped_results[ds_display]
            samples = ds_group.get("Sequencial", {}).get("samples", 0)
            
            it_seq = ds_group.get("Sequencial", {}).get("iterations", "N/A")
            it_mpi = ds_group.get("MPI + OpenMP", {}).get("iterations", "N/A")
            it_omp_gpu = ds_group.get("OpenMP GPU", {}).get("iterations", "N/A")
            it_cuda = ds_group.get("CUDA", {}).get("iterations", "N/A")
            
            f.write(f"| **{ds_display}** | {samples:,} | {it_seq} | {it_mpi} | {it_omp_gpu} | {it_cuda} |\n")
            
        f.write("\n---\n\n")
        
        # Detalhes específicos de CUDA (tempo de cópia de memória)
        f.write("## Detalhes Adicionais de Cópia de Memória (CUDA)\n\n")
        f.write("| Dataset | Qtd. Amostras | Tempo Total (s) | Host-to-Device (ms) | Device-to-Host (ms) | Custo Transf. Total (s) | % Overhead Transf. |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for ds_display in [d["display"] for d in DATASETS]:
            if ds_display not in grouped_results:
                continue
            ds_group = grouped_results[ds_display]
            cuda_r = ds_group.get("CUDA", {})
            t_cuda = cuda_r.get("time")
            extras = cuda_r.get("extras", {})
            
            if t_cuda is not None and 'h2d_ms' in extras and 'd2h_ms' in extras:
                h2d = extras['h2d_ms']
                d2h = extras['d2h_ms']
                transf_total_s = (h2d + d2h) / 1000.0
                pct_overhead = (transf_total_s / t_cuda) * 100.0
                f.write(f"| **{ds_display}** | {cuda_r['samples']:,} | {t_cuda:.6f} | {h2d:.2f} | {d2h:.2f} | {transf_total_s:.6f} | {pct_overhead:.2f}% |\n")
            else:
                f.write(f"| **{ds_display}** | {cuda_r.get('samples', 0):,} | N/A | N/A | N/A | N/A | N/A |\n")

    log_success(f"Arquivo Markdown consolidado gerado: {md_file}")

if __name__ == "__main__":
    main()
