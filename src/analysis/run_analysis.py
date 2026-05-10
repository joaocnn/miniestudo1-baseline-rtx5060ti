"""
Orquestrador da análise pós-coleta.

Uso (CLI):
    python -m src.analysis.run_analysis --csv data/raw/<arquivo>.csv

Input:
    --csv  Caminho para o CSV bruto gerado por gemm_benchmark.py.
           Aceita caminho absoluto ou relativo à raiz do projeto.

Outputs:
    data/processed/summary_statistics.csv  — estatísticas por N (tempo_ms)
    data/processed/outliers_log.txt        — outliers IQR registrados (não removidos)
    figures/boxplot_tempo_por_N.png
    figures/gflops_vs_N.png
    figures/temperatura_durante_coleta.png
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from src.analysis.plots import (
    plot_boxplot_tempo_por_N,
    plot_gflops_vs_N,
    plot_temperatura_durante_coleta,
)
from src.analysis.stats import compute_summary, detect_outliers_iqr

PROJECT_ROOT = Path(__file__).parents[2]

EXPECTED_COLUMNS = {"run", "cenario", "N", "tempo_ms", "gflops", "gpu_temp_c", "timestamp", "observacao"}


def _load_and_validate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        print(f"ERRO: colunas ausentes no CSV: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)
    df["N"] = df["N"].astype(int)
    df["tempo_ms"] = df["tempo_ms"].astype(float)
    df["gflops"] = df["gflops"].astype(float)
    df["gpu_temp_c"] = pd.to_numeric(df["gpu_temp_c"], errors="coerce").fillna(0).astype(int)
    return df


def _detect_and_log_outliers(df: pd.DataFrame, log_path: Path, csv_name: str) -> int:
    lines = []
    for N_val, group in df.groupby("N", sort=True):
        mask = detect_outliers_iqr(group["tempo_ms"].to_numpy())
        for idx, is_outlier in zip(group.index, mask):
            if is_outlier:
                row = group.loc[idx]
                lines.append(
                    f"run={int(row['run']):<4}  N={N_val:<5}  tempo_ms={row['tempo_ms']:.4f}  motivo=IQR"
                )

    with log_path.open("w") as f:
        f.write(f"# Outliers detectados pelo critério IQR (k=1.5) — {csv_name}\n")
        f.write(f"# Total: {len(lines)}\n\n")
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("(nenhum outlier detectado)\n")

    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise dos dados brutos do benchmark GEMM.")
    parser.add_argument("--csv", required=True, help="Caminho para o CSV bruto de medições.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        print(f"ERRO: arquivo não encontrado: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = _load_and_validate(csv_path)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = processed_dir / "summary_statistics.csv"
    outliers_log_path = processed_dir / "outliers_log.txt"

    # Resumo estatístico de tempo_ms
    summary = compute_summary(df, group_col="N", value_col="tempo_ms")
    summary.to_csv(summary_path, index=False, float_format="%.4f")

    # Detecção de outliers (registra, não remove)
    n_outliers = _detect_and_log_outliers(df, outliers_log_path, csv_path.name)

    # Figuras
    plot_boxplot_tempo_por_N(df, figures_dir / "boxplot_tempo_por_N.png")
    plot_gflops_vs_N(df, figures_dir / "gflops_vs_N.png")
    plot_temperatura_durante_coleta(df, figures_dir / "temperatura_durante_coleta.png")

    # Resumo no terminal
    display = summary.rename(
        columns={
            "N": "N",
            "n": "n",
            "mean": "média (ms)",
            "median": "mediana (ms)",
            "min": "mín (ms)",
            "max": "máx (ms)",
            "std": "dp",
            "ci95_low": "IC95 inf",
            "ci95_high": "IC95 sup",
        }
    )
    print("\n=== Resumo Estatístico — tempo_ms por N ===\n")
    print(
        tabulate(
            display.values.tolist(),
            headers=display.columns.tolist(),
            tablefmt="rounded_outline",
            floatfmt=".3f",
        )
    )

    print(f"\n[Outliers] {n_outliers} encontrado(s) — ver {outliers_log_path.relative_to(PROJECT_ROOT)}")
    print("\n[Saídas geradas]")
    for path in (
        summary_path,
        outliers_log_path,
        figures_dir / "boxplot_tempo_por_N.png",
        figures_dir / "gflops_vs_N.png",
        figures_dir / "temperatura_durante_coleta.png",
    ):
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
