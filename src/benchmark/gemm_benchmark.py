"""
Benchmark principal de GEMM denso (float32) na RTX 5060 Ti via cuBLAS/CuPy.

Uso:
    python -m src.benchmark.gemm_benchmark --timestamp YYYYMMDD_HHMMSS

Saída:
    data/raw/gemm_rtx5060ti_<timestamp>.csv

Cada linha do CSV corresponde a uma medição individual; warmups
não são gravados. Instrumento de medição: cp.cuda.Event.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cupy as cp

from src.benchmark import config
from src.benchmark.gpu_utils import (
    assert_single_gpu,
    check_gpu_idle,
    get_gpu_clock_mhz,
    get_gpu_compute_pids,
    get_gpu_info,
    get_gpu_temperature,
)

# Raiz do projeto (dois níveis acima de src/benchmark/)
PROJECT_ROOT = Path(__file__).parents[2]


def _resolve_output_path(timestamp_str: str) -> Path:
    """Monta e valida o caminho do CSV de saída."""
    out_dir = PROJECT_ROOT / config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gemm_rtx5060ti_{timestamp_str}.csv"
    if path.exists():
        print(f"ERRO: arquivo de saída já existe: {path}", file=sys.stderr)
        print("Abortar para não sobrescrever dados.", file=sys.stderr)
        sys.exit(1)
    return path


def _print_header(gpu_info: dict, output_path: Path) -> None:
    """Imprime cabeçalho informativo antes da confirmação."""
    print("=" * 60)
    print("BENCHMARK GEMM — RTX 5060 Ti — Baseline ICC305")
    print("=" * 60)
    print("\n[GPU]")
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")
    print(f"  clock_mhz: {get_gpu_clock_mhz()}")
    print(f"  temp_c: {get_gpu_temperature()}")
    print("\n[Experimento]")
    print(f"  cenario:      {config.SCENARIO_NAME}")
    print(f"  dtype:        {config.DTYPE}")
    print(f"  sizes:        {config.SIZES}")
    print(f"  repetitions:  {config.REPETITIONS}")
    print(f"  warmup_runs:  {config.WARMUP_RUNS}")
    print(f"  seed:         {config.SEED}")
    print(f"\n[Saída]")
    print(f"  {output_path}")

    # Estimativa grosseira: ~1 ms por iteração para N=1024 (ajuste heurístico)
    total_iters = len(config.SIZES) * (config.WARMUP_RUNS + config.REPETITIONS)
    est_sec = total_iters * 0.005  # 5 ms por iter é conservador
    print(f"\n[Duração estimada] ~{est_sec:.0f}s ({est_sec/60:.1f} min) — estimativa grosseira")
    print()


def _medir_iteracao(A: cp.ndarray, B: cp.ndarray) -> tuple[cp.ndarray, float]:
    """Executa cp.matmul cronometrado com CUDA Events. Retorna (C, tempo_ms)."""
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    C = cp.matmul(A, B)
    end.record()
    end.synchronize()
    tempo_ms = cp.cuda.get_elapsed_time(start, end)
    return C, tempo_ms


def run_benchmark(output_path: Path) -> None:
    """Executa a coleta completa e grava o CSV linha a linha."""
    t_inicio_total = time.monotonic()

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run", "cenario", "N", "tempo_ms", "gflops", "gpu_temp_c", "timestamp", "observacao"])
        csvfile.flush()

        for N in config.SIZES:
            print(f"\n=== Iniciando coleta para N={N} (warmup: {config.WARMUP_RUNS}, medições: {config.REPETITIONS}) ===")

            # Seed fixa antes de gerar matrizes para cada N
            cp.random.seed(config.SEED)
            A = cp.random.uniform(0.0, 1.0, (N, N), dtype=config.DTYPE)
            B = cp.random.uniform(0.0, 1.0, (N, N), dtype=config.DTYPE)
            C = cp.zeros((N, N), dtype=config.DTYPE)

            # Temperatura no início do bloco (para detectar delta_temp_alto)
            temp_inicio_bloco = get_gpu_temperature()

            # Fase de warmup — descartada
            cp.cuda.Stream.null.synchronize()
            for _ in range(config.WARMUP_RUNS):
                C, _ = _medir_iteracao(A, B)
            cp.cuda.Stream.null.synchronize()

            # Fase de medição
            linhas_bloco: list[list] = []
            for rep in range(1, config.REPETITIONS + 1):
                if rep % config.TEMP_READ_INTERVAL == 1:
                    temp_c = get_gpu_temperature()
                C, tempo_ms = _medir_iteracao(A, B)
                gflops = (2 * N**3) / (tempo_ms * 1e-3) / 1e9
                ts = datetime.now().isoformat()

                print(
                    f"[N={N}] iter {rep}/{config.REPETITIONS} | "
                    f"tempo={tempo_ms:.3f}ms | {gflops:.2f} GFLOPS | temp={temp_c}°C"
                )

                linhas_bloco.append([rep, config.SCENARIO_NAME, N, f"{tempo_ms:.6f}", f"{gflops:.4f}", temp_c, ts, "ok"])

            # Temperatura no fim do bloco
            temp_fim_bloco = get_gpu_temperature()
            delta_temp = abs(temp_fim_bloco - temp_inicio_bloco)
            observacao_bloco = "delta_temp_alto" if delta_temp > config.TEMP_DELTA_THRESHOLD else "ok"

            for linha in linhas_bloco:
                linha[-1] = observacao_bloco  # atualiza observacao
                writer.writerow(linha)
                csvfile.flush()

            # Libera memória antes do próximo N
            del A, B, C
            cp.get_default_memory_pool().free_all_blocks()

    t_fim_total = time.monotonic()
    duracao = t_fim_total - t_inicio_total
    minutos = int(duracao // 60)
    segundos = duracao % 60

    print("\nColeta concluída.")
    print(f"  Arquivo: {output_path}")
    print(f"  Total de medições: {config.REPETITIONS * len(config.SIZES)}")
    print(f"  Tempo total: {minutos}min {segundos:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GEMM float32 na RTX 5060 Ti via cuBLAS/CuPy.")
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp no formato YYYYMMDD_HHMMSS para nomear o CSV de saída.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Pular confirmação interativa (útil ao chamar via script).",
    )
    args = parser.parse_args()

    timestamp_str = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = _resolve_output_path(timestamp_str)

    # Verificações pré-coleta
    assert_single_gpu()

    if not check_gpu_idle():
        pids = get_gpu_compute_pids()
        print(
            f"ERRO: a GPU não está ociosa — {len(pids)} processo(s) compute ativo(s): PIDs {pids}",
            file=sys.stderr,
        )
        print(
            "Encerre esses processos antes de prosseguir (use 'kill <PID>' ou 'nvidia-smi' para detalhes).",
            file=sys.stderr,
        )
        sys.exit(1)

    gpu_info = get_gpu_info()
    _print_header(gpu_info, output_path)

    if not args.no_confirm:
        resposta = input('Digite "yes" para iniciar a coleta (qualquer outra coisa aborta): ')
        if resposta.strip() != "yes":
            print("Coleta cancelada pelo usuário.")
            sys.exit(0)

    run_benchmark(output_path)


if __name__ == "__main__":
    main()
