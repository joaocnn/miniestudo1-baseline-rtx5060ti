# Mini Estudo 1 — Baseline RTX 5060 Ti

Caracterização de desempenho de multiplicação de matrizes densas (GEMM, float32)
na GPU NVIDIA GeForce RTX 5060 Ti 16 GB, via cuBLAS, para tamanhos de matriz
N ∈ {256, 512, 1024, 2048, 4096}.

**Disciplina:** ICC305 — Avaliação de Desempenho  
**Contexto:** Mini Estudo 1 — coleta do baseline de SGEMM denso

Equipe:
Caio Aleixo Cunha

João Carlos Normando Nogueira

Leonardo de Souza Bitencourt Irias

---

## Pergunta operacional

> Qual é o desempenho-base (em GFLOPS) e o tempo de execução (em ms) da
> multiplicação de matrizes quadradas densas em float32 na RTX 5060 Ti
> 16 GB, utilizando cuBLAS, para tamanhos N ∈ {256, 512, 1024, 2048, 4096}?

## Sistema avaliado

- **GPU:** NVIDIA GeForce RTX 5060 Ti 16 GB (Blackwell, sm_120)
- **Operação:** `C = A · B`, matrizes N×N em float32 via `cp.matmul` (cuBLAS SGEMM)
- **Ambiente:** máquina dedicada, GPU exclusiva (sem compartilhamento), modo de
  energia máximo desempenho, driver e CUDA fixados e documentados em
  `data/raw/metadata_<timestamp>.json`

## Pré-requisitos

- Ubuntu Linux (testado em 24.04)
- Driver NVIDIA recente compatível com Blackwell (série 500+)
- CUDA Toolkit 12.x
- Python 3.10+
- `mamba` ou `conda` (para criar o ambiente)
- GPU NVIDIA RTX 5060 Ti 16 GB exclusivamente disponível

## Como reproduzir

```bash
# 1. Clonar o repositório
git clone <url>
cd miniestudo1-baseline-rtx5060ti

# 2. Criar e ativar o ambiente conda/mamba
mamba env create -f environment.yml
mamba activate avd-miniestudo1

# 3. Rodar o pipeline completo (verificação + metadados + benchmark + análise)
bash scripts/run_experiment.sh
```

O script `run_experiment.sh` executa todas as etapas em sequência e aborta se
qualquer uma falhar. Os artefatos aparecem em `data/raw/`, `data/processed/` e
`figures/` com o mesmo timestamp.

## Estrutura do repositório

```
miniestudo1-baseline-rtx5060ti/
├── docs/
│   ├── Relatorio_MiniEstudo1_Baseline_RTX5060Ti.pdf
├── src/
│   ├── benchmark/
│   │   ├── config.py           # Parâmetros do experimento (N, reps, seed…)
│   │   ├── gemm_benchmark.py   # Coleta: CUDA Events + cuBLAS via CuPy
│   │   └── gpu_utils.py        # Wrappers nvidia-smi e CuPy
│   └── analysis/
│       ├── stats.py            # Funções estatísticas puras (IC, IQR, resumo)
│       ├── plots.py            # Geração de figuras (3 plots)
│       └── run_analysis.py     # Orquestrador da análise pós-coleta
├── scripts/
│   ├── verify_environment.sh   # Checa nvidia-smi, python3, CuPy, CUDA
│   ├── collect_metadata.sh     # Coleta JSON de metadados do sistema
│   └── run_experiment.sh       # Pipeline completo (1 comando)
├── data/
│   ├── raw/                    # CSVs brutos + JSONs de metadados
│   └── processed/              # Resumo estatístico + log de outliers
├── figures/                    # PNGs gerados pela análise
├── notebooks/                  # Análise exploratória (opcional)
├── environment.yml             # Ambiente conda reprodutível
└── requirements.txt            # Dependências pip
```

## Onde encontrar os resultados

| Artefato | Caminho |
|---|---|
| Dados brutos | `data/raw/gemm_rtx5060ti_<timestamp>.csv` |
| Metadados da coleta | `data/raw/metadata_<timestamp>.json` |
| Resumo estatístico | `data/processed/summary_statistics.csv` |
| Outliers identificados | `data/processed/outliers_log.txt` |
| Boxplot tempo por N | `figures/boxplot_tempo_por_N.png` |
| GFLOPS vs N | `figures/gflops_vs_N.png` |
| Temperatura durante coleta | `figures/temperatura_durante_coleta.png` |
| Relatório | `docs/Relatorio_MiniEstudo1_Baseline_RTX5060Ti.pdf` |

## Limitações

Este baseline **não permite concluir**:

- Que essa GPU é melhor ou pior que qualquer outra
- O desempenho em outras precisões (fp16, bf16, fp64, int8)
- O desempenho em GEMM batched, esparso, ou com matrizes não-quadradas
- O desempenho em workloads reais (LLM, treino, etc.)
- Comportamento sob carga térmica sustentada (>30 min)
- Comportamento em outras versões de driver, CUDA ou cuBLAS

## Contexto acadêmico

Trabalho desenvolvido para a disciplina ICC305 — Avaliação de Desempenho.
Este miniestudo é o primeiro de uma série de experimentos sobre caracterização
de desempenho de GPUs em workloads de multiplicação de matrizes densas.

Para citar este repositório, use o hash do commit que gerou os dados ou o
timestamp registrado no arquivo de metadados (`data/raw/metadata_<timestamp>.json`).
