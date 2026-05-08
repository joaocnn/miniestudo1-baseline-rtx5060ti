"""
Configuração do experimento de baseline GEMM na RTX 5060 Ti.

Todos os parâmetros do experimento ficam centralizados aqui
para garantir reprodutibilidade e facilitar ajustes pontuais
(ex.: piloto com SIZES menor antes da coleta final).
"""

# Tamanhos das matrizes quadradas N x N a serem testados.
# Cobrem desde GPU subutilizada até GPU saturada.
SIZES = [256, 512, 1024, 2048, 4096]

# Repetições medidas por configuração de N.
# Mínimo de 30 conforme exigido pela atividade (Seção 5 do enunciado).
REPETITIONS = 30

# Iterações descartadas antes da medição, para estabilizar
# clocks da GPU (GPU Boost) e cache de instruções.
WARMUP_RUNS = 5

# Seed para gerador de números aleatórios das matrizes A e B.
# Fixar a seed garante que a mesma carga é aplicada em todas
# as repetições e em re-execuções futuras.
SEED = 42

# Tipo de dado das matrizes. Float32 é o padrão para SGEMM.
DTYPE = "float32"

# Diretório de saída dos CSVs brutos (relativo à raiz do projeto).
OUTPUT_DIR = "data/raw"

# Identificador do cenário, usado na coluna 'cenario' do CSV.
SCENARIO_NAME = "rtx5060ti_cublas_fp32"
