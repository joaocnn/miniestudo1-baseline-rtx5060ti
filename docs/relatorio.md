# Mini Estudo 1 — Baseline RTX 5060 Ti

**Disciplina:** ICC305 — Avaliação de Desempenho  
**Autor:** João  
**Data:** [a preencher após a coleta]  
**Commit / timestamp da coleta:** [a preencher]

---

## 1. Pergunta operacional

[a preencher após a coleta]

<!-- 
  Copiar verbatim da ficha_planejamento.md, item 1:
  "Qual é o desempenho-base (em GFLOPS) e o tempo de execução (em ms) da
   multiplicação de matrizes quadradas densas em float32 na RTX 5060 Ti
   16 GB, utilizando cuBLAS, para tamanhos N ∈ {256, 512, 1024, 2048, 4096}?"
  
  Também indicar o que esta pergunta NÃO cobre (ver item 15 da ficha),
  para deixar claro o escopo do estudo.
-->

## 2. Sistema avaliado

[a preencher após a coleta]

<!-- 
  Descrever a configuração exata usada na coleta, extraída do
  data/raw/metadata_<timestamp>.json:
  - GPU: modelo, memória, compute capability
  - Driver NVIDIA e versão do CUDA runtime
  - Versão do cuBLAS
  - CPU, RAM total
  - Sistema operacional e versão do kernel
  - Modo de energia da GPU (persistence mode)
  - Confirmar que nenhum outro processo usava a GPU durante a coleta
  
  Colar a saída de nvidia-smi do momento da coleta (ou referenciá-la
  no Apêndice A).
-->

## 3. Métrica e instrumento

[a preencher após a coleta]

<!-- 
  Métrica principal: tempo de execução do kernel GEMM em milissegundos (ms).
  Métrica derivada: throughput em GFLOPS = 2·N³ / (tempo_s · 10⁹).
  
  Instrumento: CUDA Events (cp.cuda.Event), que medem o tempo decorrido
  diretamente na GPU com resolução de microssegundos. Justificar por que
  timers de host (time.perf_counter) seriam inadequados aqui (overhead de
  lançamento de kernel e sincronização host↔device).
  
  Confirmar que cópias host↔device ficam fora da região cronometrada.
-->

## 4. Benchmark e carga de trabalho

[a preencher após a coleta]

<!-- 
  Descrever o microbenchmark:
  - Operação: C = A · B, matrizes N×N float32
  - Chamada via cp.matmul (cuBLAS SGEMM por baixo dos panos)
  - N ∈ {256, 512, 1024, 2048, 4096}
  - Valores aleatórios uniformes [0, 1), seed 42 (reprodutível)
  - Footprint por N: 3·N²·4 bytes (A, B, C em float32)
    - N=256 ≈ 0,75 MB; N=4096 ≈ 192 MB (bem abaixo dos 16 GB)
  - FLOPs por execução: 2·N³ (contagem padrão SGEMM)
  - Padrão de acesso: regular, contíguo (favorável ao cuBLAS)
  
  Discutir por que este microbenchmark é representativo (ou não) de
  workloads reais.
-->

## 5. Protocolo de coleta

[a preencher após a coleta]

<!-- 
  Descrever o procedimento executado por run_experiment.sh:
  1. Verificação do ambiente (scripts/verify_environment.sh)
  2. Coleta de metadados do sistema (scripts/collect_metadata.sh)
  3. Para cada N: 5 warm-up runs descartados + 30 medições registradas
  4. Temperatura lida a cada 5 iterações; flag "delta_temp_alto" se ΔT > 5 °C
  5. Memória GPU liberada entre configurações de N
  
  Mencionar condições de controle:
  - GPU exclusiva (verificada por nvidia-smi antes da coleta)
  - Driver, CUDA e cuBLAS fixados (documentados no JSON de metadados)
  - Execução sequencial (sem paralelismo)
  - Ausência de carga pesada de CPU em paralelo
  - Seed fixa garante matrizes idênticas entre re-execuções
-->

## 6. Resultados

### 6.1 Resumo estatístico

[a preencher após a coleta]

<!-- 
  Colar ou referenciar a tabela de data/processed/summary_statistics.csv.
  Colunas: N, n, mean, median, min, max, std, ci95_low, ci95_high (tempo_ms).
  
  Comentar:
  - O desvio padrão relativo (CV = std/mean) para cada N — indica estabilidade
  - Se os IC 95% são estreitos (boa repetibilidade) ou largos (alta variância)
  - Algum N com comportamento atípico?
-->

### 6.2 Distribuição dos tempos (boxplot)

[a preencher após a coleta]

<!-- 
  Inserir a figura figures/boxplot_tempo_por_N.png.
  
  Comentar:
  - Escala log no eixo y — justificar (tempos variam ~1000× entre N=256 e N=4096)
  - Simetria ou assimetria de cada distribuição
  - Presença ou ausência de outliers visuais
  - Largura dos boxes (variabilidade intra-N)
-->

### 6.3 Throughput em GFLOPS

[a preencher após a coleta]

<!-- 
  Inserir a figura figures/gflops_vs_N.png.
  
  Comentar:
  - Tendência geral: GFLOPS sobe com N até saturar (platô de roofline)?
  - Em qual N o cuBLAS começa a explorar bem o hardware?
  - Barras de erro (IC 95%): são estreitas? Indicam medições estáveis?
  - GFLOPS pico teórico da RTX 5060 Ti em FP32: ~XX TFLOPS
    (consultar ficha técnica NVIDIA e comparar com o medido)
  - Eficiência alcançada (% do pico teórico) para cada N
-->

### 6.4 Verificação da ameaça térmica

[a preencher após a coleta]

<!-- 
  Inserir a figura figures/temperatura_durante_coleta.png.
  
  Comentar:
  - Temperatura inicial e final de cada bloco de N
  - Houve blocos com flag "delta_temp_alto" (ΔT > 5 °C)?
    Se sim, quais N foram afetados e como isso se reflete no desvio padrão?
  - A temperatura se estabilizou dentro de cada bloco ou continuou subindo?
  - Conclusão: ameaça térmica foi controlada ou precisa de revisão do protocolo?
-->

## 7. Interpretação

[a preencher após a coleta]

<!-- 
  Integrar as observações das subseções 6.1–6.4 em uma narrativa coesa:
  
  - Qual é o GFLOPS característico desta GPU para SGEMM denso nesta faixa de N?
  - Em que N o cuBLAS atinge o melhor aproveitamento do hardware?
  - Há evidência de throttling térmico ou de clock? Se sim, em qual N?
  - As medições são repetíveis o suficiente para servir de referência futura?
  - O que este baseline diz sobre a capacidade da GPU para workloads de IA/HPC?
  
  Manter o escopo: este estudo NÃO permite comparar com outras GPUs.
-->

## 8. Limitações

[a preencher após a coleta]

<!-- 
  Copiar do ficha_planejamento.md, item 15, e comentar cada ponto
  à luz dos dados coletados:
  
  - Que essa GPU é melhor ou pior que qualquer outra (sem comparativo)
  - O desempenho em outras precisões (fp16, bf16, fp64, int8)
  - O desempenho em GEMM batched, esparso, ou com matrizes não-quadradas
  - O desempenho em workloads reais (LLM, treino, etc.)
  - Comportamento sob carga térmica sustentada (>30 min)
  - Comportamento em outras versões de driver, CUDA ou cuBLAS
  
  Discutir também se alguma limitação metodológica foi identificada
  durante ou após a coleta (ex.: warm-up insuficiente, variabilidade
  inesperada, dados faltantes).
-->

## 9. Conclusão

[a preencher após a coleta]

<!-- 
  Síntese de 1–2 parágrafos respondendo diretamente à pergunta operacional:
  
  Parágrafo 1: resultado quantitativo central — o GFLOPS característico
  para N suficientemente grande, com IC 95%, e o tempo mediano para cada N.
  
  Parágrafo 2: o que este baseline habilita — comparações futuras na mesma GPU,
  com diferentes drivers/precisões, ou como ponto de referência para estudos
  subsequentes da disciplina.
-->

---

## Apêndice A — Metadados da coleta

[a preencher após a coleta]

<!-- 
  Colar (ou referenciar) o conteúdo de data/raw/metadata_<timestamp>.json,
  formatado como tabela markdown ou bloco de código JSON.
  
  Campos relevantes para o leitor:
  - GPU: nome, driver, VBIOS, memória total, clock base/boost
  - CUDA runtime e cuBLAS
  - CPU, RAM
  - OS, kernel
  - Python, CuPy, NumPy, pandas, scipy
  - Timestamp da coleta
-->

## Apêndice B — Outliers identificados

[a preencher após a coleta]

<!-- 
  Colar o conteúdo de data/processed/outliers_log.txt.
  
  Para cada outlier, comentar:
  - Qual run e N
  - Se coincide com leitura de temperatura elevada (flag "delta_temp_alto")
  - Se foi a primeira ou última iteração de um bloco (clocks ainda instáveis?)
  - Decisão: manter nos dados (não excluir) — justificar com base em que
    outliers por IQR com n=30 podem ser flutuações normais de GPU Boost
-->
