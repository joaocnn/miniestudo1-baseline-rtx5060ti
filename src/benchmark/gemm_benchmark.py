"""
Benchmark principal de GEMM denso (float32) na RTX 5060 Ti via cuBLAS/CuPy.

Uso:
    python -m src.benchmark.gemm_benchmark --timestamp YYYYMMDD_HHMMSS

Saída:
    data/raw/gemm_rtx5060ti_<timestamp>.csv

Cada linha do CSV corresponde a uma medição individual; warmups
não são gravados. Instrumento de medição: cp.cuda.Event.
"""

# TODO: implementar no Prompt 2
pass
