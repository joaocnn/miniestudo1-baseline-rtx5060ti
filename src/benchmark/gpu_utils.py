"""
Utilitários para inspeção e controle da GPU NVIDIA.

Uso (import):
    from src.benchmark.gpu_utils import get_gpu_info, check_gpu_idle, ...

Wrappers sobre nvidia-smi (subprocess) e CuPy para:
- Ler temperatura, clock e uso da GPU
- Verificar que a GPU está ociosa antes da coleta
- Garantir que exatamente uma GPU está visível

Nenhuma função aqui modifica estado da GPU ou inicia kernels CUDA.
"""

import subprocess
import cupy as cp


def _nvidia_smi_query(query: str, format_opts: str = "csv,noheader,nounits") -> str:
    """Executa nvidia-smi com --query-gpu e retorna stdout bruto."""
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", f"--format={format_opts}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_gpu_temperature() -> int:
    """Retorna a temperatura atual da GPU em graus Celsius."""
    return int(_nvidia_smi_query("temperature.gpu"))


def get_gpu_clock_mhz() -> int:
    """Retorna o clock atual do SM da GPU em MHz."""
    return int(_nvidia_smi_query("clocks.gr"))


def get_gpu_compute_pids() -> list[int]:
    """Retorna lista de PIDs dos processos compute ativos na GPU."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    return [int(pid) for pid in lines]


def get_gpu_processes_count() -> int:
    """Retorna o número de processos compute ativos na GPU."""
    return len(get_gpu_compute_pids())


def check_gpu_idle() -> bool:
    """Retorna True se não há processos compute ativos na GPU."""
    return get_gpu_processes_count() == 0


def assert_single_gpu() -> None:
    """Aborta se o número de GPUs visíveis for diferente de 1."""
    count = cp.cuda.runtime.getDeviceCount()
    if count != 1:
        raise RuntimeError(
            f"Esperava exatamente 1 GPU visível, mas encontrou {count}. "
            "Use CUDA_VISIBLE_DEVICES para selecionar uma única GPU antes de executar."
        )


def get_gpu_info() -> dict:
    """Retorna dicionário com metadados estáticos da GPU."""
    queries = {
        "name": "name",
        "driver_version": "driver_version",
        "vbios": "vbios_version",
        "memory_total_mb": "memory.total",
    }
    info = {}
    for key, q in queries.items():
        raw = _nvidia_smi_query(q)
        if key == "memory_total_mb":
            # nvidia-smi retorna "X MiB" quando sem --nounits;
            # com nounits já devolve o número.
            info[key] = int(raw)
        else:
            info[key] = raw

    # compute_capability via CuPy — string "XY", ex.: "89" → "8.9"
    cc = cp.cuda.Device(0).compute_capability  # '120'
    # Para Blackwell: 12.0; para Ampere ('86'): 8.6
    info["compute_capability"] = f"{cc[:-1]}.{cc[-1]}" if len(cc) >= 2 else cc

    return info
