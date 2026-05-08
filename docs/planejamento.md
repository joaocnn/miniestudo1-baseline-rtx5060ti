# Ficha de Planejamento — Mini Estudo 1

**Disciplina:** ICC305 — Avaliação de Desempenho
**Trilha:** Algoritmos e matrizes (GEMM denso)
**Sistema analisado:** NVIDIA GeForce RTX 5060 Ti 16 GB

---

## 1. Pergunta operacional

Qual é o desempenho-base (em GFLOPS) e o tempo de execução (em ms) da
multiplicação de matrizes quadradas densas em float32 na RTX 5060 Ti
16 GB, utilizando cuBLAS, para tamanhos N ∈ {256, 512, 1024, 2048, 4096}?

## 2. Sistema / cenário-base

GPU NVIDIA RTX 5060 Ti 16 GB, em uma única máquina hospedeira, sem
compartilhamento da GPU com outros processos, em modo de energia
"máximo desempenho", com o driver e o CUDA Toolkit fixados em uma
versão específica documentada.

## 3. Métrica principal e unidade

Tempo de execução do kernel GEMM, em **milissegundos (ms)**.
Métrica derivada (reportada junto): throughput em **GFLOPS**, calculado
como `2·N³ / (tempo_em_segundos · 10⁹)`.

Tempo é a grandeza medida diretamente; GFLOPS é uma transformação
determinística dela.

## 4. Instrumento de medição

CUDA Events (`cp.cuda.Event` no CuPy, equivalente a `cudaEventRecord`
+ `cudaEventElapsedTime`), que medem o tempo decorrido na própria GPU,
com resolução de microssegundos.

Não usar timers do host (`time.time`, `time.perf_counter`) — eles
incluem overhead de lançamento de kernel e sincronização.

## 5. O instrumento mede diretamente a métrica?

Diretamente. CUDA Events medem o tempo decorrido entre dois pontos do
stream da GPU. A métrica derivada (GFLOPS) assume que o GEMM executa
exatamente `2·N³` operações de ponto flutuante — contagem padrão para
SGEMM denso na literatura.

## 6. Benchmark / microbenchmark

Microbenchmark chamando diretamente cuBLAS (via `cp.matmul` do CuPy),
sem framework intermediário. Reduz camadas de abstração que poderiam
introduzir variabilidade.

## 7. Carga de trabalho aplicada

Multiplicação `C = A · B`, com:
- A, B, C matrizes quadradas N×N
- Tipo: float32
- N ∈ {256, 512, 1024, 2048, 4096}
- Valores: aleatórios uniformes em [0, 1), seed fixa (42)
- Memória alocada na GPU; cópias host↔device fora da região cronometrada

## 8. Caracterização da carga

- Densidade: 100% (matrizes densas, sem zeros estruturais)
- Distribuição dos valores: uniforme em [0, 1)
- Footprint por configuração: 3·N²·4 bytes (A, B, C em float32).
  Para N=4096 ≈ 192 MB — bem abaixo dos 16 GB da GPU.
- FLOPs por execução: 2·N³
- Padrão de acesso: regular, contíguo

## 9. Repetições

30 repetições medidas por configuração de N (5 tamanhos × 30 = 150
medições). Antes das medidas, 5 warm-up runs descartados por
configuração para estabilizar clocks da GPU.

## 10. O que será mantido constante

- Driver NVIDIA e versão do CUDA Toolkit
- Versão do cuBLAS (via CuPy)
- Modo de energia da GPU (persistence mode ON)
- Sistema operacional e versão do kernel
- Ausência de outros processos usando a GPU (verificado via nvidia-smi)
- Ausência de carga pesada de CPU em paralelo
- Temperatura ambiente (registrada no início e fim)
- Mesma seed para geração das matrizes
- Execução sequencial das repetições

## 11. Dado bruto registrado

CSV com uma linha por repetição individual:
run, cenario, N, tempo_ms, gflops, gpu_temp_c, timestamp, observacao

## 12. Metadados registrados

Arquivo JSON separado contendo:
- GPU: modelo, driver, VBIOS, memória total
- Versão do CUDA Toolkit e do cuBLAS
- CPU, RAM, motherboard
- Sistema operacional + versão do kernel
- Versões de Python, CuPy, NumPy, pandas, scipy
- Saída completa de nvidia-smi
- Data e hora de início e fim da coleta

## 13. Análise mínima

- Resumo estatístico por N: média, mediana, mínimo, máximo, desvio padrão
- Intervalo de confiança 95% da média (t-Student, gl=29)
- Boxplot de tempo_ms por N
- GFLOPS médio vs N com barras de erro (IC 95%)
- Identificação de outliers por IQR (registrados, não excluídos)

## 14. O que o baseline permitirá concluir

- O desempenho-base da RTX 5060 Ti 16 GB para SGEMM denso via cuBLAS,
  na faixa de N testada, neste ambiente
- Como o desempenho escala com N nessa GPU
- A variabilidade das medições nesta configuração
- Um ponto de referência para comparar com mudanças futuras na mesma GPU

## 15. O que o baseline NÃO permitirá concluir

- Que essa GPU é melhor ou pior que qualquer outra
- O desempenho em outras precisões (fp16, bf16, fp64, int8)
- O desempenho em GEMM batched, esparso, ou com matrizes não-quadradas
- O desempenho em workloads reais (LLM, treino, etc.)
- Comportamento sob carga térmica sustentada (>30 min)
- Comportamento em outras versões de driver, CUDA ou cuBLAS

## 16. Principal ameaça à validade

Variabilidade térmica e de clock dinâmico (GPU Boost). A RTX 5060 Ti
ajusta clocks em função de temperatura e potência. Sem warm-up
adequado, repetições iniciais rodam em clocks diferentes das finais,
inflando o desvio padrão e enviesando a média.

## 17. Cuidado metodológico principal

Warm-up disciplinado e monitoramento térmico. Antes de cada
configuração de N, executar 5 iterações descartadas. Registrar
temperatura via nvidia-smi antes e depois de cada bloco de 30
repetições. Variações de temperatura >5 °C dentro de um bloco são
sinalizadas como observação no dado bruto.