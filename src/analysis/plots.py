"""
Geração de figuras a partir do DataFrame de medições.

Uso (import):
    from src.analysis.plots import plot_boxplot_tempo_por_N, plot_gflops_vs_N, ...

Input:  DataFrame pandas com as colunas do CSV bruto do benchmark.
Output: Arquivos PNG em figures/ (dpi=150, bbox_inches='tight').

Figuras produzidas:
- Boxplot de tempo_ms por N (escala log no eixo y)
- GFLOPS médio vs N com barras de erro IC 95% (escala log₂ no eixo x)
- Temperatura da GPU ao longo da coleta com índice global e delimitadores de bloco
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from src.analysis.stats import compute_summary


def _save(fig: plt.Figure, output_path) -> None:
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_tempo_por_N(df: pd.DataFrame, output_path) -> None:
    """Boxplot de tempo_ms agrupado por N; eixo y em escala log."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    order = sorted(df["N"].unique())
    sns.boxplot(
        data=df,
        x="N",
        y="tempo_ms",
        order=order,
        ax=ax,
        hue="N",
        palette="Blues_d",
        linewidth=0.8,
        flierprops={"marker": ".", "markersize": 4, "alpha": 0.6},
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_title("Distribuição do Tempo de Execução por Tamanho de Matriz", fontsize=13)
    ax.set_xlabel("Tamanho da Matriz (N)", fontsize=11)
    ax.set_ylabel("Tempo (ms) — escala log", fontsize=11)

    _save(fig, output_path)


def plot_gflops_vs_N(df: pd.DataFrame, output_path) -> None:
    """Linha de GFLOPS médio por N com barras de erro (IC 95%); eixo x em log₂."""
    sns.set_theme(style="whitegrid")

    summary = compute_summary(df, group_col="N", value_col="gflops")
    ns = summary["N"].to_numpy()
    means = summary["mean"].to_numpy()
    err_low = means - summary["ci95_low"].to_numpy()
    err_high = summary["ci95_high"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        ns,
        means,
        yerr=[err_low, err_high],
        marker="o",
        linewidth=2,
        capsize=5,
        color="steelblue",
        ecolor="gray",
        label="GFLOPS médio ± IC 95%",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_title("GFLOPS Médio por Tamanho de Matriz", fontsize=13)
    ax.set_xlabel("Tamanho da Matriz (N) — escala log₂", fontsize=11)
    ax.set_ylabel("GFLOPS", fontsize=11)
    ax.legend(fontsize=10)

    _save(fig, output_path)


def plot_temperatura_durante_coleta(df: pd.DataFrame, output_path) -> None:
    """Temperatura ao longo das medições com índice global contínuo e separadores por bloco de N."""
    sns.set_theme(style="whitegrid")

    groups = list(df.groupby("N", sort=True))
    palette = sns.color_palette("tab10", n_colors=len(groups))

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plota linhas e acumula os offsets de cada bloco
    block_info: list[tuple[int, int, int]] = []  # (start, size, N_val)
    cursor = 0
    for color, (N_val, group) in zip(palette, groups):
        n = len(group)
        ax.plot(
            range(cursor, cursor + n),
            group["gpu_temp_c"].values,
            label=f"N={N_val}",
            color=color,
            linewidth=1.2,
        )
        block_info.append((cursor, n, N_val))
        cursor += n

    # Linhas verticais tracejadas separando os blocos
    for start, _size, _N_val in block_info[1:]:
        ax.axvline(x=start, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Anotações indicando onde começa cada N (topo interno da figura)
    xaxis_transform = ax.get_xaxis_transform()  # x=dados, y=fração do eixo
    for color, (start, size, N_val) in zip(palette, block_info):
        ax.text(
            start + size / 2,
            0.97,
            f"N={N_val}",
            transform=xaxis_transform,
            ha="center",
            va="top",
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )

    ax.set_title("Temperatura da GPU Durante a Coleta", fontsize=13)
    ax.set_xlabel("Índice da Medição (global, 0–149)", fontsize=11)
    ax.set_ylabel("Temperatura (°C)", fontsize=11)
    ax.legend(title="Tamanho N", fontsize=9, title_fontsize=9, loc="lower right")

    _save(fig, output_path)
