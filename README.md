# Mini Estudo 1 — Baseline RTX 5060 Ti

Caracterização de desempenho de multiplicação de matrizes densas
(GEMM, float32) na GPU NVIDIA GeForce RTX 5060 Ti 16 GB, via cuBLAS.

**Disciplina:** ICC305 — Avaliação de Desempenho

---

## Pergunta operacional

Qual é o desempenho-base (em GFLOPS) e o tempo de execução (em ms) da
multiplicação de matrizes quadradas densas em float32 na RTX 5060 Ti
16 GB, utilizando cuBLAS, para tamanhos N ∈ {256, 512, 1024, 2048, 4096}?

## Pré-requisitos

- Ubuntu Linux (testado em 24.04)
- Driver NVIDIA recente (compatível com Blackwell, série 5000)
- CUDA Toolkit 12.x
- Python 3.10+
- GPU NVIDIA RTX 5060 Ti 16 GB

## Como reproduzir

```bash
# 1. Clonar o repositório
git clone <url>
cd miniestudo1-baseline-rtx5060ti

# 2. Criar ambiente virtual e instalar dependências
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Verificar o ambiente
bash scripts/verify_environment.sh

# 4. Rodar o experimento completo (coleta + análise)
bash scripts/run_experiment.sh
```

Resultados aparecem em `data/raw/`, `data/processed/` e `figures/`.

## Estrutura do repositório
miniestudo1-baseline-rtx5060ti/
├── docs/              # Ficha de planejamento e relatório
├── src/
│   ├── benchmark/     # Coleta de dados (CuPy + cuBLAS)
│   └── analysis/      # Estatística e figuras
├── scripts/           # Pipeline shell (verify, metadata, run)
├── data/
│   ├── raw/           # CSVs brutos + metadados JSON (uma por execução)
│   └── processed/     # Resumo estatístico e log de outliers
├── figures/           # Gráficos gerados
└── notebooks/         # Análise exploratória (opcional)

## Onde encontrar os resultados

| Tipo | Caminho |
|---|---|
| Dados brutos | `data/raw/gemm_rtx5060ti_<timestamp>.csv` |
| Metadados da coleta | `data/raw/metadata_<timestamp>.json` |
| Resumo estatístico | `data/processed/summary_statistics.csv` |
| Outliers identificados | `data/processed/outliers_log.txt` |
| Figuras | `figures/*.png` |
| Relatório | `docs/relatorio.md` (e versão final em PDF) |

## Limitações

Este baseline **não permite concluir**:

- Que esta GPU é melhor ou pior que outras
- Desempenho em outras precisões (fp16, bf16, fp64, int8)
- Desempenho em GEMM batched, esparso ou com matrizes não-quadradas
- Comportamento em workloads reais (LLM, treino, etc.)
- Comportamento sob carga térmica sustentada (>30 min)
- Comportamento em outras versões de driver, CUDA ou cuBLAS

Veja `docs/ficha_planejamento.md`, item 15, para a lista completa.

## Contexto acadêmico

Trabalho desenvolvido para a disciplina ICC305 — Avaliação de
Desempenho. Este miniestudo é o primeiro de uma série de experimentos
sobre caracterização de desempenho de GPUs em workloads de
multiplicação de matrizes densas.