"""
Funções estatísticas puras para análise dos dados brutos.

Uso (import):
    from src.analysis.stats import compute_summary, detect_outliers_iqr, confidence_interval

Inputs:  DataFrame pandas com as colunas do CSV bruto gerado pelo benchmark.
Outputs: DataFrames e arrays numpy — sem efeitos colaterais (sem I/O).

Inclui resumo descritivo (média, mediana, min, max, desvio padrão),
intervalo de confiança 95% via t-Student (gl = n-1) e detecção de
outliers pelo critério IQR.
"""

import numpy as np
import pandas as pd
from scipy import stats


def confidence_interval(values, confidence: float = 0.95) -> tuple[float, float]:
    """IC via t-Student bicaudal (gl = n-1). Retorna (low, high)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    se = stats.sem(arr)
    h = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    mean = np.mean(arr)
    return mean - h, mean + h


def detect_outliers_iqr(values, k: float = 1.5) -> np.ndarray:
    """Bool array — True nas posições fora de [Q1 - k*IQR, Q3 + k*IQR]."""
    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return (arr < q1 - k * iqr) | (arr > q3 + k * iqr)


def compute_summary(
    df: pd.DataFrame,
    group_col: str = "N",
    value_col: str = "tempo_ms",
) -> pd.DataFrame:
    """
    Resumo estatístico por grupo.

    Retorna DataFrame com colunas:
        <group_col>, n, mean, median, min, max, std, ci95_low, ci95_high
    """
    rows = []
    for group_val, group_df in df.groupby(group_col, sort=True):
        vals = group_df[value_col].to_numpy(dtype=float)
        n = len(vals)
        ci_low, ci_high = confidence_interval(vals)
        rows.append(
            {
                group_col: group_val,
                "n": n,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "std": float(np.std(vals, ddof=1)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)
